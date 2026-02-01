# Copyright (c) 2026, Signal Processing Group. University of Hamburg. All rights reserved.
#
# Code adapted from https://github.com/NVlabs/edm2
# 
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop."""

import os
import time
import copy
import pickle
import psutil
import torch
import dnnlib
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from os.path import join, exists
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc

from torch.utils.tensorboard import SummaryWriter
from generate import generate


#----------------------------------------------------------------------------

class EDM2SELoss:
    def __init__(self, 
        k=2.6, 
        c=0.4, 
        c_s=1,
        sigma_x=0.402, 
        sigma_n=0.342, 
        eps=1e-8, 
        t_eps=0.02, 
        T=1.0, 
        l1_weight=0.001, 
        audio_encoder=None, 
        weight=True
    ):
        self.k = torch.tensor([k], device='cuda')
        self.c = torch.tensor([c], device='cuda')
        self.c_s = c_s
        self.sigma_x = torch.tensor([sigma_x], device='cuda')
        self.sigma_n = torch.tensor([sigma_n], device='cuda')
        self.eps = torch.tensor([eps], device='cuda')
        self.t_eps = torch.tensor([t_eps], device='cuda')
        self.T = torch.tensor([T], device='cuda')
        self.l1_weight = l1_weight
        self.audio_encoder = audio_encoder
        self.weight = weight

    def alpha(self, t):
        alpha = torch.ones_like(t)
        return alpha

    def sigma(self, t):
        sigma = torch.sqrt((self.c * (self.k ** (2 * t) - 1.0)) / (2 * torch.log(self.k)) + self.eps)
        return sigma

    def alpha_bar(self, t):
        alpha_bar = torch.ones_like(t)
        return alpha_bar

    def sigma_bar(self, t):
        sigma_bar = torch.sqrt(self.sigma(self.T) ** 2 - self.sigma(t) ** 2 + self.eps)
        return sigma_bar

    def w_x(self, t):
        w_x = (self.alpha(t) * self.sigma_bar(t) **2) / (self.sigma(self.T) ** 2 + self.eps)             
        return w_x

    def w_y(self, t):
        w_y = (self.alpha_bar(t) * self.sigma(t) ** 2) / (self.sigma(self.T) ** 2 + self.eps)              
        return w_y

    def mean(self, x0, y, t):
        w_x = self.w_x(t)  
        w_y = self.w_y(t)            
        mu = w_x * x0 + w_y * y     
        return mu

    def std(self, t):
        std = (self.alpha(t) * self.sigma_bar(t) * self.sigma(t)) / (self.sigma(self.T) + self.eps)
        return std

    def __call__(self, net, audio, spkr_embeds=None, cond_feats=None):
        t = torch.rand([audio.shape[0], 1, 1, 1], device=audio.device) * (self.T - self.t_eps) + self.t_eps
        B, C, F, T = audio.shape
        L = (F - 1) * self.audio_encoder.hop_length
        std = self.std(t)
        mean = self.mean(audio, cond_feats, t)
        w_x = self.w_x(t)
        w_y = self.w_y(t)

        # Compute loss weight based on net.c_s (Eqs. 24 and 25)
        if self.c_s == 1:
            weight = (1 / (torch.sqrt((1 - w_x - w_y) ** 2 
                * self.sigma_x ** 2 + w_y ** 2 * self.sigma_n ** 2 + std ** 2)))
        elif self.c_s == 0:
            weight = 1 / self.sigma_x ** 2
        else:
            raise ValueError(
                f"Invalid value for net.c_s: {net.c_s}. "
                "Expected net.c_s to be either 0 or 1."
            )

        # Sample Gaussian noise.
        noise = std * torch.randn_like(audio)
        
        # Compute loss in the time-frequency domain
        denoised, logvar = net(mean + noise, t, spkr_embeds, cond_feats, return_logvar=True)
        loss_tf = torch.mean((denoised - audio) ** 2, dim=(1, 2, 3)) 

        # Compute loss in the time domain
        denoised_time = self.audio_encoder.decode(denoised)
        audio_time = self.audio_encoder.decode(audio)
        loss_time = torch.mean(torch.abs(denoised_time - audio_time), dim=1)

        # Combine time-frequency and time domain losses
        unscaled_loss = loss_tf + self.l1_weight * loss_time
        loss = (weight / (logvar.exp() + self.eps)) * unscaled_loss + logvar

        loss_dict = dnnlib.EasyDict(
            sigma=std, 
            weight=weight, 
            logvar=logvar, 
            unscaled_loss=unscaled_loss, 
            loss_tf=loss_tf, 
            loss_time=loss_time
        )

        return loss, loss_dict

#----------------------------------------------------------------------------
# Learning rate decay schedule 

def learning_rate_schedule(cur_nwav, batch_size, ref_lr=100e-4, ref_batches=70e3, rampup_mwav=10):
    lr = ref_lr
    if ref_batches > 0:
        lr /= np.sqrt(max(cur_nwav / (ref_batches * batch_size), 1))
    if rampup_mwav > 0:
        lr *= min(cur_nwav / (rampup_mwav * 1e6), 1)
    return lr

#----------------------------------------------------------------------------
# Main training loop.

def training_loop(
    dataset_kwargs       = dict(class_name='training.dataset_se.SEDataset', path=None),
    audio_encoder_kwargs = dict(class_name='training.encoders.SpectrogramEncoder'),
    data_loader_kwargs   = dict(class_name='torch.utils.data.DataLoader', pin_memory=True, num_workers=2, prefetch_factor=2),
    network_kwargs       = dict(class_name='training.model.EDM2SE', freq_resolution=256, spec_channels=2),
    loss_kwargs          = dict(class_name='training.training_loop.EDM2Loss'),
    optimizer_kwargs     = dict(class_name='torch.optim.Adam', betas=(0.9, 0.99)),
    lr_kwargs            = dict(func_name='training.training_loop.learning_rate_schedule'),
    ema_kwargs           = dict(class_name='training.phema.PowerFunctionEMA'),

    run_dir              = '.',      # Output directory.
    seed                 = 0,        # Global random seed.
    batch_size           = 32,       # Total batch size for one training iteration.
    batch_gpu            = None,     # Limit batch size per GPU. None = no limit.
    total_nwav           = 1024<<20, # Train for a total of N training audio.
    slice_nwav           = None,     # Train for a maximum of N training audio in one invocation. None = no limit.
    status_nwav          = 128<<10,  # Report status every N training audio. None = disable.
    snapshot_nwav        = 1024<<10, # Save network snapshot every N training audio. None = disable.
    checkpoint_nwav      = 2048<<10, # Save state checkpoint every N training audio. None = disable.

    loss_scaling         = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    force_finite         = True,     # Get rid of NaN/Inf gradients before feeding them to the optimizer.
    cudnn_benchmark      = True,     # Enable torch.backends.cudnn.benchmark?
    device               = torch.device('cuda'),
    noval                = False,    # Disable validation   
    num_files            = None,     # Number of files to validate
    verbose              = False,    # Print more information
):
    # Initialize.
    prev_status_time = time.time()
    misc.set_random_seed(seed, dist.get_rank())
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    writer = SummaryWriter(join(run_dir, "logs"))

    # Validate batch size.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()
    assert total_nwav % batch_size == 0
    assert slice_nwav is None or slice_nwav % batch_size == 0
    assert status_nwav is None or status_nwav % batch_size == 0
    assert snapshot_nwav is None or (snapshot_nwav % batch_size == 0) # and snapshot_nwav % 1024 == 0)
    assert checkpoint_nwav is None or (checkpoint_nwav % batch_size == 0) # and checkpoint_nwav % 1024 == 0)

    # Setup dataset, encoder, and network.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    audio_encoder = dnnlib.util.construct_class_by_name(**audio_encoder_kwargs)

    dist.print0('Constructing network...')
    net = dnnlib.util.construct_class_by_name(**network_kwargs)
    net.train().requires_grad_(True).to(device)

    # Check if validation dir exist.
    validation_path = join(dataset_obj.path, "valid", "noisy")
    if not exists(validation_path) and not noval:
        raise FileNotFoundError(
            f"Validation path does not exist: {validation_path}"
    )
    
    if verbose and dist.get_rank() == 0:
        misc.print_module_summary(net, [ #x, noise_labels, spk_embed, cond_feats
            torch.zeros([batch_gpu, net.spec_channels, net.freq_resolution, 256], device=device),
            torch.ones([batch_gpu], device=device) if 'L2Loss' not in loss_kwargs.class_name else None,
            None,
            torch.zeros([batch_gpu, net.spec_channels, net.freq_resolution, 256], device=device) if 'L2Loss' not in loss_kwargs.class_name else None,
        ], max_nesting=2)

    # Setup training state.
    state = dnnlib.EasyDict(cur_nwav=0, total_elapsed_time=0)
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], find_unused_parameters=True)
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs, audio_encoder=audio_encoder)
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs)
    ema = dnnlib.util.construct_class_by_name(net=net, **ema_kwargs) if ema_kwargs is not None or 'L2Loss' not in loss_kwargs.class_name else None

    # Load previous checkpoint.
    checkpoint = dist.CheckpointIO(state=state, net=net, loss_fn=loss_fn, optimizer=optimizer, ema=ema)
    checkpoint.load_latest(run_dir)

    # Decide when to stop.
    stop_at_nwav = total_nwav
    if slice_nwav is not None:
        granularity = checkpoint_nwav if checkpoint_nwav is not None else snapshot_nwav if snapshot_nwav is not None else batch_size
        slice_end_nwav = (state.cur_nwav + slice_nwav) // granularity * granularity # round down
        stop_at_nwav = min(stop_at_nwav, slice_end_nwav)
    assert stop_at_nwav > state.cur_nwav

    dist.print0("Starting training loop...")
    # Main training loop.
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed, start_idx=state.cur_nwav)
    dataset_iterator = iter(dnnlib.util.construct_class_by_name(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, collate_fn=dataset_obj.collate_fn, **data_loader_kwargs))
    prev_status_nwav = state.cur_nwav
    cumulative_training_time = 0
    start_nwav = state.cur_nwav
    stats_jsonl = None

    pbar = tqdm(total=stop_at_nwav, initial=state.cur_nwav, unit='wavs', unit_scale=True, dynamic_ncols=True, disable=(dist.get_rank() != 0))
    while True:
        done = (state.cur_nwav >= stop_at_nwav)

        # Report status.
        if status_nwav is not None and (done or state.cur_nwav % status_nwav == 0) and (state.cur_nwav != start_nwav or start_nwav == 0):
            cur_time = time.time()
            state.total_elapsed_time += cur_time - prev_status_time
            cur_process = psutil.Process(os.getpid())
            cpu_memory_usage = sum(p.memory_info().rss for p in [cur_process] + cur_process.children(recursive=True))
            dist.print0()
            dist.print0(' '.join(['Status:',
                'kwav', f"{training_stats.report0('Progress/kwav', state.cur_nwav / 1e3):.1f}",
                '\ttime', f"{dnnlib.util.format_time(training_stats.report0('Timing/total_sec', state.total_elapsed_time)):s}",
                '\tsec/tick', f"{training_stats.report0('Timing/sec_per_tick', cur_time - prev_status_time):.2f}",
                '\tsec/kwav', f"{training_stats.report0('Timing/sec_per_kwav', cumulative_training_time / max(state.cur_nwav - prev_status_nwav, 1) * 1e3):.3f}",
                '\tmaintenance', f"{training_stats.report0('Timing/maintenance_sec', cur_time - prev_status_time - cumulative_training_time):.2f}",
                '\tcpumem', f"{training_stats.report0('Resources/cpu_mem_gb', cpu_memory_usage / 2**30):.2f}",
                '\tgpumem', f"{training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):.2f}",
                '\treserved', f"{training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):.2f}",
            ]))
            cumulative_training_time = 0
            prev_status_nwav = state.cur_nwav
            prev_status_time = cur_time
            torch.cuda.reset_peak_memory_stats()

            # Flush training stats.
            training_stats.default_collector.update()
            if dist.get_rank() == 0:
                if stats_jsonl is None:
                    stats_jsonl = open(join(run_dir, 'stats.jsonl'), 'at')
                fmt = {'Progress/tick': '%.0f', 'Progress/kwav': '%.3f', 'timestamp': '%.3f'}
                items = [(name, value.mean) for name, value in training_stats.default_collector.as_dict().items()] + [('timestamp', time.time())]
                items = [f'"{name}": ' + (fmt.get(name, '%g') % value if np.isfinite(value) else 'NaN') for name, value in items]
                stats_jsonl.write('{' + ', '.join(items) + '}\n')
                stats_jsonl.flush()

            # Update progress and check for abort.
            dist.update_progress(state.cur_nwav // 1000, stop_at_nwav // 1000)
            if state.cur_nwav == stop_at_nwav and state.cur_nwav < total_nwav:
                dist.request_suspend()
            if dist.should_stop() or dist.should_suspend():
                done = True

        # Save network snapshot.
        if snapshot_nwav is not None and state.cur_nwav % snapshot_nwav == 0 and (state.cur_nwav != start_nwav or start_nwav == 0):
            if dist.get_rank() == 0:
                ema_list = ema.get() if ema is not None else optimizer.get_ema(net) if hasattr(optimizer, 'get_ema') else net
                ema_list = ema_list if isinstance(ema_list, list) else [(ema_list, '')]
                for ema_net, ema_suffix in ema_list:
                    data = dnnlib.EasyDict(audio_encoder=audio_encoder, dataset_kwargs=dataset_kwargs, loss_fn=loss_fn)
                    data.ema = copy.deepcopy(ema_net).cpu().eval().requires_grad_(False).to(torch.float16)
                    fname = f'network-snapshot-{state.cur_nwav//1000:07d}{ema_suffix}.pkl'
                    with open(join(run_dir, fname), 'wb') as f:
                        pickle.dump(data, f)
                    del data # conserve memory
            if validation_path is not None and not noval:
                net.eval()
                with misc.switch_backend("valid"), torch.no_grad():
                    # Generate audio samples
                    dist.print0(f"\nGenerating audio samples for state {state.cur_nwav}...")
                    audio_iter = generate(
                        net=net,
                        audio_encoder=audio_encoder,
                        out_dir=join(run_dir, "valid", f'{state.cur_nwav}'),
                        test_dir=validation_path,
                        num_files=num_files,
                    )
                    # Loop over batches.
                    for _r in tqdm(audio_iter, total=len(audio_iter)):
                        pass
            
                net.train()
                dist.print0(f"\nContinue training loop...")

        # Save state checkpoint.
        if checkpoint_nwav is not None and (done or state.cur_nwav % checkpoint_nwav == 0) and state.cur_nwav != start_nwav:
            checkpoint.save(join(run_dir, f'training-state-{state.cur_nwav//1000:07d}.pt'))
            # delete previous checkpoint
            checkpoint.delete_first(run_dir)
            misc.check_ddp_consistency(net)

        # Done?
        if done:
            break

        # Evaluate loss and accumulate gradients.
        batch_start_time = time.time()
        misc.set_random_seed(seed, dist.get_rank(), state.cur_nwav)
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                item = next(dataset_iterator)
                clean_audio = item['clean_audio']
                noisy_audio = item['noisy_audio']
                
                # Encode audio.
                clean_audio = audio_encoder.encode(clean_audio.to(device))
                noisy_audio = audio_encoder.encode(noisy_audio.to(device))

                loss, loss_dict = loss_fn(net=ddp, audio=clean_audio, spkr_embeds=None, cond_feats=noisy_audio)
                training_stats.report('Loss/loss', loss)
                final_loss = loss.sum().mul(loss_scaling / batch_gpu_total)
                final_loss.backward()

                writer.add_scalar('Loss/train_loss', final_loss, state.cur_nwav)
                writer.add_scalar('Loss/loss_tf', loss_dict.loss_tf.sum()/ batch_gpu_total, state.cur_nwav)
                writer.add_scalar('Loss/loss_time', loss_dict.loss_time.sum()/ batch_gpu_total, state.cur_nwav)
                writer.add_scalar('Loss/unscaled_loss', loss_dict.unscaled_loss.sum()/ batch_gpu_total, state.cur_nwav)
                writer.add_scalar('Loss/logvar', loss_dict.logvar.sum()/ batch_gpu_total, state.cur_nwav)

        # Run optimizer and update weights.
        lr = dnnlib.util.call_func_by_name(cur_nwav=state.cur_nwav, batch_size=batch_size, **lr_kwargs)
        training_stats.report('Loss/learning_rate', lr)

        writer.add_scalar('Loss/LR', lr, state.cur_nwav)

        for g in optimizer.param_groups:
            g['lr'] = lr
        if force_finite:
            for param in net.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)
        optimizer.step()

        # Update EMA and training state.
        state.cur_nwav += batch_size
        if ema is not None:
            ema.update(cur_nwav=state.cur_nwav, batch_size=batch_size)
        cumulative_training_time += time.time() - batch_start_time
        pbar.update(batch_size)
    pbar.close()
    writer.close()

#----------------------------------------------------------------------------
