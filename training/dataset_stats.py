# Copyright (c) 2026, Signal Processing Group. University of Hamburg. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Calculate dataset statistics."""

import click
import torch
import dnnlib
import torchaudio 
import numpy as np

from glob import glob
from tqdm import tqdm
from os.path import join
from training.encoders import SpectrogramEncoder


#----------------------------------------------------------------------------

def calculate_stats(opts):
    # Audio paths.
    clean_paths = sorted(
        glob(join(opts['data'], opts['subset'], 'clean', '**', '*.wav'), recursive=True)
    )
    noisy_paths = sorted(
        glob(join(opts['data'], opts['subset'], 'noisy', '**', '*.wav'), recursive=True)
    )
    print(f"Found {len(clean_paths)} clean files and {len(noisy_paths)} noisy files.")

    # Set up encoder.
    encoder = SpectrogramEncoder(raw_std=0.08, final_std=0.5)
    target_len = (opts['num_frames'] - 1) * opts['hop_length']

    # Fix seed for reproducability. 
    np.random.seed(0)

    print("Generating spectrograms...")
    x_std_list = []
    n_std_list = []

    for clean_path, noisy_path in tqdm(zip(clean_paths, noisy_paths), total=len(clean_paths)):
        # Load the audio files.
        x, _ = torchaudio.load(clean_path)
        y, _ = torchaudio.load(noisy_path)

        # Take only the first channel.
        x = x[0]
        y = y[0]

        # Normalize audio to [-1, 1].
        normfac = y.abs().max()
        x = x / normfac
        y = y / normfac

        # Calculete the noise.
        n = y - x

        current_len = x.size(-1)
        pad = max(target_len - current_len, 0)
        if pad == 0:
            # Extract random part of the audio file.
            start = int(np.random.uniform(0, current_len-target_len))
            x = x[start:start + target_len]
            n = n[start:start + target_len]
        else:
            # Pad audio if the length is smaller than num_frames.
            x = torch.nn.functional.pad(x, (pad//2, pad//2+(pad%2)), mode='constant')
            n = torch.nn.functional.pad(n, (pad//2, pad//2+(pad%2)), mode='constant')

        x_spec = encoder.encode(x.unsqueeze(0))
        n_spec = encoder.encode(n.unsqueeze(0))

        x_std_list.append(x_spec.std().cpu().numpy())
        n_std_list.append(n_spec.std().cpu().numpy())

    print("Concatenating...")
    x_std_values = np.stack(x_std_list)
    n_std_values = np.stack(n_std_list)

    x_std = np.mean(x_std_values) 
    n_std = np.mean(n_std_values)

    print(f"sigma_x: {x_std:.3f}, sigma_n: {n_std:.3f}")

#----------------------------------------------------------------------------
# Command line interface.

@click.command()
@click.option('--data',       help='Path to the dataset',                 type=str, required=True)
@click.option('--subset',     help='Train, valid, or test set?',          type=str, default='train')
@click.option('--num_frames', help='Number of frames in the spectrogram', type=int, default=256)
@click.option('--hop_length', help='Hop lengths for the STFT',            type=int, default=128)

def cmdline(**opts):
    """Calculate dataset statistics based on the training data.

    Example usage:

    Make sure to add the project root to PYTHONPATH.

    python training/dataset_stats.py \
        --data /path/to/data_dir 
    """

    opts = dnnlib.EasyDict(opts)
    calculate_stats(opts)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------
