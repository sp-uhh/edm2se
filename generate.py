# Copyright (c) 2026, Signal Processing Group. University of Hamburg. All rights reserved.
#
# Code adapted from https://github.com/NVlabs/edm2
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate audio conditioned on noisy features using the given model."""

import os
import torch
import click
import dnnlib
import torchaudio
import numpy as np
import soundfile as sf

from tqdm import tqdm
from glob import glob
from torch_utils import distributed as dist
from training.model import EDM2SE
from training.encoders import SpectrogramEncoder

#----------------------------------------------------------------------------
# SB-ODE sampler from the paper "Schrödinger bridge for generative speech enhancement"

def sb_sampler(
    net, noise, cond=None, num_steps=50, t_eps=0.02, T=1, k=2.6, c=0.4, 
    eps=1e-8, sampler="ode", **kwargs
): 
    def alpha(t):
        alpha = torch.ones_like(t)
        return alpha

    def sigma(t):
        sigma = torch.sqrt((c * (k ** (2 * t) - 1.0)) / (2 * torch.log(torch.tensor([k], device=noise.device))) + eps)
        return sigma

    def sigma_bar(t):
        sigma_bar = torch.sqrt(sigma(T) ** 2 - sigma(t) ** 2 + eps)
        return sigma_bar

    time_steps = torch.linspace(T, 0, num_steps + 1, device=noise.device)
    xt = cond

    # Initial values
    T = T * torch.ones(xt.shape[0], device=xt.device)
    time_prev = time_steps[0] * torch.ones(xt.shape[0], device=xt.device)
    sigma_prev = sigma(time_prev)
    sigma_bar_prev = sigma_bar(time_prev)
    sigma_T = sigma(T)
    alpha_prev = alpha(time_prev)
    alpha_T = alpha(T)

    for t in time_steps[1:]:
        # Prepare time steps for the whole batch
        t = t * torch.ones(xt.shape[0], device=xt.device)
        alpha_cur = alpha(t)
        sigma_cur = sigma(t)
        sigma_bar_cur = sigma_bar(t)

        # Run DNN
        D = net(xt, time_prev, cond)

        # Calculate scaling for the first-order discretization from the paper
        if sampler == "ode":
            weight_prev = alpha_cur * sigma_cur * sigma_bar_cur / (alpha_prev * sigma_prev * sigma_bar_prev + eps)
            weight_estimate = (
                alpha_cur
                / (sigma_T**2 + eps)
                * (sigma_bar_cur**2 - sigma_bar_prev * sigma_cur * sigma_bar_cur / (sigma_prev + eps))
            )
            weight_prior_mean = (
                alpha_cur
                / (alpha_T * sigma_T**2 + eps)
                * (sigma_cur**2 - sigma_prev * sigma_cur * sigma_bar_cur / (sigma_bar_prev + eps))
            )
            weight_z = 0.0
        elif sampler == "sde":
            weight_prev = alpha_cur * sigma_cur ** 2 / (alpha_prev * sigma_prev ** 2 + eps)
            weight_estimate = alpha_cur * (1 - sigma_cur ** 2 / (sigma_prev ** 2 + eps))
            weight_prior_mean = 0.0
            weight_z = alpha_cur * sigma_cur * torch.sqrt(1 - sigma_cur ** 2 / (sigma_prev ** 2 + eps))
        else:
            raise ValueError(
                f"Sampler should either 'ode' or 'sde', but got '{sampler}'"
            )
            
        # Random sample
        z = torch.randn_like(xt)

        # Update state: weighted sum of previous state, current estimate and prior
        xt = weight_prev * xt + weight_estimate * D + weight_prior_mean * cond + weight_z * z

        # Save previous values
        time_prev = t
        alpha_prev = alpha_cur
        sigma_prev = sigma_cur
        sigma_bar_prev = sigma_bar_cur

    return D
    
#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(size=size[1:], generator=gen, **kwargs) for gen in self.generators])
    
    def randint_like(self, input, **kwargs):
        return self.randint(input.shape, dtype=input.dtype, layout=input.layout, device=input.device, **kwargs)


#----------------------------------------------------------------------------
# Generate audio 

def generate(
    net,                                      # Main network. Path, or torch.nn.Module.
    audio_encoder     = None,                 # Instance of training.encoders.Encoder. None = load from network pickle.
    out_dir           = None,                 # Where to save the output images. None = do not save.
    seed              = 0,                    # Seed for the random number generator.
    test_dir          = None,                 # Path to the noisy files.
    verbose           = False,                # Enable status prints?
    device            = torch.device('cuda'), # Which compute device to use.
    num_files         = None,                 # Number of files to generate.
    sampler           = "ode",                # Sampling type, either "ode" or "sde"
    **sampler_kwargs,                         # Additional arguments for the sampler function.
):
    # Build the model
    if isinstance(net, str):
        assert os.path.isfile(net), f"Checkpoint not found: {net}"
        ckpt = torch.load(net, map_location="cpu", weights_only=False)
        net = EDM2SE()
        net.load_state_dict(ckpt)
        net = net.to(device)
    elif isinstance(net, torch.nn.Module):
        net = net.to(device)
    else:
        raise TypeError(
            f"`net` must be a str (checkpoint path) or torch.nn.Module, got {type(net)}"
        )

    # Build the audio encoder
    if audio_encoder is None:
        audio_encoder = SpectrogramEncoder()

    # Retrieve the conditional audio paths
    cond_paths = sorted(glob(os.path.join(test_dir, '**', '*.wav'), recursive=True))

    # Randmly select num_files with based on seed 
    if num_files is not None:
        np.random.seed(seed)
        cond_paths = np.random.choice(cond_paths, num_files, replace=False)
    
    seeds = len(cond_paths) * [seed]

    # Divide seeds into batches.
    if dist.get_world_size() == 1:
        rank_batches = [i for i in range(len(seeds))]
    else:
        num_batches = max((len(seeds) - 1) // dist.get_world_size() + 1, 1) * dist.get_world_size()
        rank_batches = np.array_split(np.arange(len(seeds)), num_batches)[dist.get_rank() :: dist.get_world_size()]

    if verbose:
        dist.print0(f'Generating {len(seeds)} audio signals...')
    
    # Return an iterable over the batches.
    class AudioIterable:
        def __len__(self):
            return len(rank_batches)

        def __iter__(self):
            # Loop over batches.
            for batch_idx, idx in enumerate(rank_batches):
                idx = int(idx)
                r = dnnlib.EasyDict(
                    audio=None, 
                    cond_feats=None,
                    cond_feats_path=cond_paths[idx],
                    noise=None,
                    batch_idx=batch_idx, 
                    num_batches=len(rank_batches), 
                    index=idx
                )
                r.seed = seeds[idx]

                y, sr_y = torchaudio.load(cond_paths[idx])
                assert sr_y == audio_encoder.sr

                # Take only the first channel.
                y = y[0]

                # Normalize and store length
                normfac = y.abs().max()
                y = y / normfac
                y_len = y.size(-1)

                # Convert to spectrogram
                y = audio_encoder.encode(y.unsqueeze(0))
                og_spec_length = y.shape[-1]

                # Pad the spectrogram length to be divisible by 2**len(net.unet.cblock).
                if hasattr(net, 'unet'):
                    gcd = 2**len(net.unet.cblock)
                else:
                    gcd = 2**len(net.cblock)
                num_pad = gcd - og_spec_length % gcd if og_spec_length % gcd != 0 else 0
                spec_length = og_spec_length + num_pad
                y = torch.nn.ReflectionPad2d((0, num_pad, 0,0))(y)
                r.cond_feats = y.to(device)

                with torch.no_grad():
                    # Pick noise and labels.
                    rnd = StackedRandomGenerator(device, [r.seed])
                    r.noise = rnd.randn([1, net.spec_channels, net.freq_resolution, spec_length], device=device)
        
                    latents = dnnlib.util.call_func_by_name(func_name=sb_sampler, net=net, noise=r.noise,
                        cond=r.cond_feats, sampler=sampler, **sampler_kwargs)
                    r.audio = audio_encoder.decode(latents[..., :og_spec_length])

                # Save audio files.
                if out_dir is not None:
                    audio = r.audio.squeeze().cpu().numpy()[..., :y_len] * normfac.numpy()
                    audio_path = r.cond_feats_path.replace(test_dir.rstrip('/'), out_dir.rstrip('/'))

                    # Save audio file
                    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
                    sf.write(audio_path, audio, audio_encoder.sr)

                # Yield results.
                torch.distributed.barrier(device_ids=[int(os.environ['LOCAL_RANK'])])
                yield r

    return AudioIterable()


#----------------------------------------------------------------------------
# Command line interface.

# TODO: Add options for audio and video paths
@click.command()
@click.option('--net',          help='Path to checkpoint',                   type=str,                   required=True)
@click.option('--test_dir',     help='Path to the noisy speech dir',         type=str,                   required=True)
@click.option('--out_dir',      help='Where to save the enhanced files',     type=str,                   required=True)
@click.option('--num_files',    help='Number of files to generate',          type=int,                   default=None)
@click.option('--seed',         help='List of random seeds (e.g. 1,2,5-10)', type=click.IntRange(min=0), default=0)
@click.option('--num_steps',    help='Number of sampling steps',             type=click.IntRange(min=1), default=50)
@click.option('--sampler',      help='Determine "ode" or "sde" sampling',    type=str,                   default='ode')

def cmdline(**opts):
    """Generate enhanced speech files conditioned on corrupted inputs.

    Example usage:

    python generate.py \
        --net /path/to/checkpoint.ckpt \
        --test_dir=/path/to/noisy_dir \
        --out_dir=/path/to/enhanced_dir
    """
    opts = dnnlib.EasyDict(opts)

    # Generate.
    dist.init()
    wav_iter = generate(**opts)
    for _ in tqdm(wav_iter, desc='Generating', total=len(wav_iter)):
        pass

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------
