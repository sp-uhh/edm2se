# Copyright (c) 2026, Signal Processing Group. University of Hamburg. All rights reserved.
#
# Code adapted from https://github.com/NVlabs/edm2
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Train EDM2SE according to the recipe from the paper
"Do We Need EMA for Diffusion-Based Speech Enhancement? 
Toward a Magnitude-Preserving Network Architecture"."""

import os
import json
import click
import torch
import dnnlib
from torch_utils import distributed as dist
import training.training_loop as training_loop


#----------------------------------------------------------------------------
# Configuration presets.

config_presets = {
    'edm2se': dnnlib.EasyDict(
        duration=1024<<20, 
        batch=16, 
        ref_lr=2.5e-3,
        ref_batches=3e4,
        rampup_mwav=1,
        snapshot=1024<<10,
        checkpoint=2048<<10,
        num_frames=256,
        c_s=1,
        channels=128, 
        dropout=0.00, 
        channel_mult=[1,1,2,2,2,2,2], 
        attn_resolutions=[16,]
    ),
}

#----------------------------------------------------------------------------
# Setup arguments for training.training_loop.training_loop().

def setup_training_config(preset='edm2se', **opts):
    opts = dnnlib.EasyDict(opts)
    c = dnnlib.EasyDict()
    
    # Preset.
    if preset not in config_presets:
        raise click.ClickException(f'Invalid configuration preset "{preset}"')
    for key, value in config_presets[preset].items():
        if opts.get(key, None) is None:
            opts[key] = value

    # Dataset.
    c.dataset_kwargs = dnnlib.EasyDict(
        class_name='training.dataset.Dataset', 
        path=opts.data, 
        num_frames=opts.num_frames,
    )

    # Encoder.
    c.audio_encoder_kwargs = dnnlib.EasyDict(
        class_name='training.encoders.SpectrogramEncoder'
    )

    # Hyperparameters.
    c.update(
        total_nwav=opts.duration, 
        batch_size=opts.batch,
    )

    # Network.
    c.network_kwargs = dnnlib.EasyDict(
        class_name=f"training.model.EDM2SE",
        channel_mult=opts.channel_mult,
        attn_resolutions=opts.attn_resolutions,
        c_s=opts.c_s
    )

    # Loss.
    c.loss_kwargs = dnnlib.EasyDict(
        class_name=f"training.training_loop.EDM2SELoss",
        l1_weight=opts.get('l1_weight', 0.001),
        c_s=opts.c_s
    )

    # Optimizer.
    c.lr_kwargs = dnnlib.EasyDict(
        func_name='training.training_loop.learning_rate_schedule', 
        ref_lr=opts.ref_lr, 
        ref_batches=opts.ref_batches, 
        rampup_mwav=opts.rampup_mwav
    )

    # Performance-related options.
    c.batch_gpu = opts.get('batch_gpu', 0) or None
    c.network_kwargs.use_fp16 = opts.get('fp16', False)
    c.loss_scaling = opts.get('ls', 1)
    c.cudnn_benchmark = opts.get('bench', True)

    # I/O-related options.
    c.status_nwav = opts.get('status', 0) or None
    c.snapshot_nwav = opts.get('snapshot', 0) or None
    c.checkpoint_nwav = opts.get('checkpoint', 0) or None
    c.seed = opts.get('seed', 0)
    c.noval = opts.get('noval', False)
    c.num_files = opts.get('num_files', None)
    c.verbose = opts.get('verbose', False)
    return c

#----------------------------------------------------------------------------
# Print training configuration.

def print_training_config(run_dir, c):
    dist.print0()
    dist.print0('Training config:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {run_dir}')
    dist.print0(f'Dataset path:            {c.dataset_kwargs.path}')
    dist.print0(f'Segment size:            {c.dataset_kwargs.num_frames}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'Mixed-precision:         {c.network_kwargs.use_fp16}')
    dist.print0()

#----------------------------------------------------------------------------
# Launch training.

def launch_training(run_dir, c):
    if dist.get_rank() == 0 and not os.path.isdir(run_dir):
        dist.print0('Creating output directory...')
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)

    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
    dnnlib.util.Logger(file_name=os.path.join(run_dir, 'log.txt'), file_mode='a', should_flush=True)
    training_loop.training_loop(run_dir=run_dir, **c)

#----------------------------------------------------------------------------
# Parse an integer with optional power-of-two suffix:
# 'K' = 2^10
# 'M' = 2^20
# 'G' = 2^30

def parse_nwav(s):
    if isinstance(s, int):
        return s
    if s.endswith('K'):
        return int(s[:-1]) << 10
    if s.endswith('M'):
        return int(s[:-1]) << 20
    if s.endswith('G'):
        return int(s[:-1]) << 30
    return int(s)

def parse_int_list(ctx, param, value):
    if value is None:
        return []
    try:
        return [int(x) for x in value.split(',')]
    except ValueError:
        raise click.BadParameter('Resolutions must be a comma-separated list of integers')

#----------------------------------------------------------------------------
# Command line interface.

@click.command()

# Main options.
@click.option('--run_dir',          help='Where to save the results',        type=str, required=True)
@click.option('--data',             help='Path to the dataset',              type=str, required=True)
@click.option('--preset',           help='Configuration preset',             type=str, default='edm2se', show_default=True)
@click.option('--verbose',          help='Print more information',           is_flag=True)

# Hyperparameters.
@click.option('--duration',         help='Training duration',                type=parse_nwav, default=None)
@click.option('--num_frames',       help='Segment size',                     type=parse_nwav, default=256)
@click.option('--batch',            help='Total batch size',                 type=parse_nwav, default=None)
@click.option('--channels',         help='Channel multiplier',               type=click.IntRange(min=64), default=None)
@click.option('--dropout',          help='Dropout probability',              type=click.FloatRange(min=0, max=1), default=None)
@click.option('--ref_lr',           help='Learning rate max. (alpha_ref)',   type=click.FloatRange(min=0, min_open=True), default=None)
@click.option('--ref_batches',      help='Learning rate decay (t_ref)',      type=float, default=3e4)
@click.option('--rampup_mwav',      help='Learning rate warmup',             type=int, default=1)
@click.option('--l1_weight',        help='Weight for L1 loss',               type=click.FloatRange(min=0), default=0.001)
@click.option('--logvar',           help='Use log-variance loss',            type=bool, default=True)
@click.option('--weight',           help='Use weight loss',                  type=bool, default=True)

# Performance-related options.
@click.option('--batch-gpu',        help='Limit batch size per GPU',         type=parse_nwav, default=0, show_default=True)
@click.option('--fp16',             help='Enable mixed-precision training',  type=bool, default=False, show_default=True)
@click.option('--ls',               help='Loss scaling',                     type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--bench',            help='Enable cuDNN benchmarking',        type=bool, default=True, show_default=True)

# I/O-related options.
@click.option('--status',           help='Interval of status prints',        type=parse_nwav, default=None, show_default=True)
@click.option('--snapshot',         help='Interval of network snapshots',    type=parse_nwav, default='1024K', show_default=True)
@click.option('--checkpoint',       help='Interval of training checkpoints', type=parse_nwav, default='2048K', show_default=True)
@click.option('--seed',             help='Random seed',                      type=int, default=0, show_default=True)

# Validation-related options.
@click.option('--noval',            help='Do not validate',                  is_flag=True)
@click.option('--num_files',        help='Number of files to validate',      type=int, default=None)


def cmdline(run_dir, **opts):
    """Train EDM2SE according to the recipe from the paper
    "Do We Need EMA for Diffusion-Based Speech Enhancement? 
    Toward a Magnitude-Preserving Network Architecture".

    Examples:

    \b
    # Train EMD2SE using 2 GPUs
    torchrun --standalone --nproc_per_node=2 train.py \
        --run_dir=/path/to/run_dir \
        --data=/path/to/dataset
    \b

    # To resume training, run the same command again.
    """
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    dist.print0('Setting up training config...')
    c = setup_training_config(**opts)
    if opts["verbose"]:
        print_training_config(run_dir=run_dir, c=c)
    
    launch_training(run_dir=run_dir, c=c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------
