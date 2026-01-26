# Copyright (c) 2026, Signal Processing Group. University of Hamburg. All rights reserved.
#
# Code adapted from https://github.com/NVlabs/edm2
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

import numpy as np
import torch
from torch_utils import misc

#----------------------------------------------------------------------------
# Normalize given tensor to unit magnitude with respect to the given
# dimensions. Default = all dimensions except the first.

def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)

#----------------------------------------------------------------------------
# Upsample or downsample the given tensor with the given filter,
# or keep it as is.

def resample(x, f=[1,1], mode='keep'):
    if mode == 'keep':
        return x
    f = np.float32(f)
    assert f.ndim == 1 and len(f) % 2 == 0
    pad = (len(f) - 1) // 2
    f = f / f.sum()
    f = np.outer(f, f)[np.newaxis, np.newaxis, :, :]
    f = misc.const_like(x, f)
    c = x.shape[1]
    if mode == 'down':
        return torch.nn.functional.conv2d(x, f.tile([c, 1, 1, 1]), groups=c, stride=2, padding=(pad,))
    assert mode == 'up'
    return torch.nn.functional.conv_transpose2d(x, (f * 4).tile([c, 1, 1, 1]), groups=c, stride=2, padding=(pad,))

def downsample1d(x, f=[1,1]):
    f = np.float32(f)
    assert f.ndim == 1 and len(f) % 2 == 0
    pad = (len(f) - 1) // 2
    f = (f / f.sum())[np.newaxis, np.newaxis, :]
    f = misc.const_lieke(x, f)
    c = x.shape[1]
    return torch.nn.functional.conv1d(x, f.tile([c, 1, 1]), groups=c, stride=2, padding=(pad,))

#----------------------------------------------------------------------------
# Magnitude-preserving SiLU.

def mp_silu(x):
    return torch.nn.functional.silu(x) / 0.596

#----------------------------------------------------------------------------
# Magnitude-preserving sum.

def mp_sum(a, b, t=0.5):
    if isinstance(t, torch.Tensor):
        return  a.lerp(b, t) / torch.sqrt((1 - t) ** 2 + t ** 2)
    else:
        return a.lerp(b, t) / np.sqrt((1 - t) ** 2 + t ** 2)

#----------------------------------------------------------------------------
# Magnitude-preserving concatenation.

def mp_cat(a, b, dim=1, t=0.5):
    Na = a.shape[dim]
    Nb = b.shape[dim]
    C = np.sqrt((Na + Nb) / ((1 - t) ** 2 + t ** 2))
    wa = C / np.sqrt(Na) * (1 - t)
    wb = C / np.sqrt(Nb) * t
    return torch.cat([wa * a , wb * b], dim=dim)

#----------------------------------------------------------------------------
# Magnitude-preserving Fourier features.

class MPFourier(torch.nn.Module):
    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer('freqs', 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer('phases', 2 * np.pi * torch.rand(num_channels))

    def forward(self, x):
        y = x.to(torch.float32)
        y = y.ger(self.freqs.to(torch.float32))
        y = y + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)

#----------------------------------------------------------------------------
# Magnitude-preserving convolution or fully-connected layer with force weight normalization.

class MPConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel, init=None):
        super().__init__()
        self.out_channels = out_channels
        if init == 0:
            self.weight = torch.nn.Parameter(torch.zeros(out_channels, in_channels, *kernel))
        else:
            self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, *kernel))

    def forward(self, x, gain=1):
        w = self.weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w)) # forced weight normalization
        w = normalize(w) # traditional weight normalization
        w = w * (gain / np.sqrt(w[0].numel())) # magnitude-preserving scaling
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 4
        return torch.nn.functional.conv2d(x, w, padding=(w.shape[-1]//2,))

#----------------------------------------------------------------------------
# U-Net encoder/decoder block with optional self-attention.

class Block(torch.nn.Module):
    def __init__(self,
        in_channels,                             # Number of input channels.
        out_channels,                            # Number of output channels.
        emb_channels,                            # Number of embedding channels.
        cond_feat_channels           = 2,        # Number of conditioning feature channels.
        flavor                       = 'enc',    # Flavor: 'enc' or 'dec'.
        resample_mode                = 'keep',   # Resampling: 'keep', 'up', or 'down'.
        resample_filter              = [1,1],    # Resampling filter.
        self_attention               = False,    # Include self-attention?
        channels_per_head            = 64,       # Number of channels per attention head.
        dropout                      = 0,        # Dropout probability.
        res_balance                  = 0.3,      # Balance between main branch (0) and residual branch (1).
        attn_balance                 = 0.3,      # Balance between main branch (0) and self-attention (1).
        cond_balance                 = 0.3,      # Balance between main branch (0) and conditioning branch (1).
        clip_act                     = 256,      # Clip output activations. None = do not clip.
        generative                   = True,     # Generative model (True) or predictive model (False)?
        **discarded_kwargs,                            # Additional arguments for the Block class.
    ):
        super().__init__()
        self.out_channels = out_channels
        self.flavor = flavor
        self.resample_filter = resample_filter
        self.resample_mode = resample_mode
        self.num_heads = out_channels // channels_per_head 
        self.dropout = dropout
        self.res_balance = res_balance
        self.attn_balance = attn_balance
        self.clip_act = clip_act
        self.self_attention = self_attention
        self.generative = generative
        if generative:
            self.emb_gain = torch.nn.Parameter(torch.zeros([]))
            self.emb_linear = MPConv(emb_channels, out_channels, kernel=[])
        
        self.conv_res0 = MPConv(out_channels if flavor == 'enc' else in_channels, out_channels, kernel=[3,3])
        self.conv_res1 = MPConv(out_channels, out_channels, kernel=[3,3])
        self.conv_skip = MPConv(in_channels, out_channels, kernel=[1,1]) if in_channels != out_channels else None
        self.attn_qkv = MPConv(out_channels, out_channels * 3, kernel=[1,1]) if self.num_heads != 0 else None
        self.attn_proj = MPConv(out_channels, out_channels, kernel=[1,1]) if self.num_heads != 0 else None
       
        if cond_feat_channels > 0:
            self.cond_balance = torch.nn.Parameter(cond_balance*torch.ones([]))
            if self.conv_skip is not None and flavor == 'enc':
                self.conv_cond = MPConv(cond_feat_channels, out_channels, kernel=[1,1])
            else:
                self.conv_cond = MPConv(cond_feat_channels, in_channels, kernel=[1,1])
 

    def forward(self, x, emb, cond_feats=None):
        # Main branch.

        if cond_feats is not None:
            cond_feats = resample(cond_feats, f=self.resample_filter, mode=self.resample_mode)
            cond_feats = self.conv_cond(cond_feats)
            cond_feats = normalize(cond_feats, dim=1)
        
        x = resample(x, f=self.resample_filter, mode=self.resample_mode)

        if self.flavor == 'enc':
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = normalize(x, dim=1) # pixel norm

        # Residual branch.
        y = self.conv_res0(mp_silu(x))
        if self.generative:
            c = self.emb_linear(emb, gain=self.emb_gain) + 1
            y = y * c.unsqueeze(2).unsqueeze(3).to(y.dtype)
        y = mp_silu(y)
        if self.training and self.dropout != 0:
            y = torch.nn.functional.dropout(y, p=self.dropout)
        y = self.conv_res1(y)

        # Connect the branches.
        if self.flavor == 'dec' and self.conv_skip is not None:
            x = self.conv_skip(x)
        x = mp_sum(x, y, t=self.res_balance)

        # Add conditional features
        if cond_feats is not None and self.flavor == 'enc':
            x = mp_sum(x, cond_feats, t=self.cond_balance)

        # Self-attention.
        # Note: torch.nn.functional.scaled_dot_product_attention() could be used here,
        # but we haven't done sufficient testing to verify that it produces identical results.
        if self.self_attention:
            y = self.attn_qkv(x) 
            y = y.reshape(y.shape[0], self.num_heads, -1, 3, y.shape[2] * y.shape[3]) # [bs, num_heads, c, 3, h*w]
            q, k, v = normalize(y, dim=2).unbind(3) # pixel norm & split [bs, num_heads, c, h*w]
            w = torch.einsum('nhcq,nhck->nhqk', q, k / np.sqrt(q.shape[2])).softmax(dim=3)
            y = torch.einsum('nhqk,nhck->nhcq', w, v)
            y = self.attn_proj(y.reshape(*x.shape))
            x = mp_sum(x, y, t=self.attn_balance)

        # Clip activations.
        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)
        return x

#----------------------------------------------------------------------------
# EDM2 U-Net model.

class UNet(torch.nn.Module):
    def __init__(self,
        freq_resolution,                          # Frequency resolution.
        spec_channels,                            # Spectrogram channels.
        input_channels         = 4,               # Input channels.
        output_channels        = 2,               # Output channels.
        embed_channels         = 0,               # Speaker embedding channels.
        cond_feat_channels     = 2,               # Video feature channels.
        model_channels         = 128,             # Base multiplier for the number of channels.
        channel_mult           = [1,1,2,2,2,2,2], # Per-resolution multipliers for the number of channels.
        channel_mult_noise     = None,            # Multiplier for noise embedding dimensionality. None = select based on channel_mult.
        channel_mult_emb       = None,            # Multiplier for final embedding dimensionality. None = select based on channel_mult.
        num_blocks             = 2,               # Number of residual blocks per resolution.
        attn_resolutions       = [16, 8],         # List of resolutions with self-attention.
        label_balance          = 0.5,             # Balance between noise embedding (0) and class embedding (1).
        concat_balance         = 0.5,             # Balance between skip connections (0) and main path (1).
        **block_kwargs,                           # Arguments for Block.
    ):
        super().__init__()
        cblock = [model_channels * x for x in channel_mult]
        cnoise = model_channels * channel_mult_noise if channel_mult_noise is not None else cblock[0]
        cemb = model_channels * channel_mult_emb if channel_mult_emb is not None else max(cblock)
        self.label_balance = label_balance
        self.concat_balance = concat_balance
        self.out_gain = torch.nn.Parameter(torch.zeros([]))
        self.cblock = cblock

        # Embedding.
        self.emb_fourier = MPFourier(cnoise)
        self.emb_noise = MPConv(cnoise, cemb, kernel=[])
        self.emb_label = MPConv(embed_channels, cemb, kernel=[]) if embed_channels != 0 else None

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = input_channels + 1
        for level, channels in enumerate(cblock):
            res = freq_resolution >> level
            if level == 0:
                cin = cout
                cout = channels
                self.enc[f'{res}_conv'] = MPConv(cin, cout, kernel=[3,3])
            else:
                self.enc[f'{res}_down'] = Block(
                    cout, 
                    cout, 
                    cemb, 
                    cond_feat_channels=cond_feat_channels, 
                    flavor='enc', 
                    resample_mode='down',
                    **block_kwargs
                    )
            for idx in range(num_blocks):
                cin = cout
                cout = channels
                self.enc[f'{res}_block{idx}'] = Block(
                    cin, 
                    cout, 
                    cemb, 
                    cond_feat_channels=cond_feat_channels, 
                    flavor='enc', 
                    self_attention=(res in attn_resolutions),
                    **block_kwargs
                    )

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        skips = [block.out_channels for block in self.enc.values()]
        for level, channels in reversed(list(enumerate(cblock))):
            res = freq_resolution >> level
            if level == len(cblock) - 1:
                self.dec[f'{res}_in0'] = Block(
                    cout, 
                    cout, 
                    cemb, 
                    cond_feat_channels=cond_feat_channels, 
                    flavor='dec', 
                    self_attention=True, 
                    **block_kwargs)
                self.dec[f'{res}_in1'] = Block(
                    cout, 
                    cout, 
                    cemb, 
                    cond_feat_channels=cond_feat_channels, 
                    flavor='dec', 
                    **block_kwargs)
            else:
                self.dec[f'{res}_up'] = Block(
                    cout, 
                    cout, 
                    cemb, 
                    cond_feat_channels=cond_feat_channels, 
                    flavor='dec', 
                    resample_mode='up',
                    **block_kwargs
                    )
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = channels
                self.dec[f'{res}_block{idx}'] = Block(
                    cin, 
                    cout, 
                    cemb, 
                    cond_feat_channels=cond_feat_channels, 
                    flavor='dec', 
                    self_attention=(res in attn_resolutions),
                    **block_kwargs
                    )
        self.out_conv = MPConv(cout, output_channels, kernel=[3,3])

    def forward(self, x, noise_labels, spk_embed, cond_feats=None):
        # Embedding.
        emb = self.emb_noise(self.emb_fourier(noise_labels))
        if self.emb_label is not None:
            emb = mp_sum(emb, self.emb_label(spk_embed * np.sqrt(spk_embed.shape[1])), t=self.label_balance)
        emb = mp_silu(emb)
        
        #-------------------------------------------------------------------
        # Encoder.
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        skips = []

        if cond_feats is not None:
            cond_feats_down = [cond_feats]
        # Iterate through encoder blocks.
        for name, block in self.enc.items():
            x = block(x) if 'conv' in name else block(x, emb, cond_feats)
            skips.append(x)
            if 'down' in name and cond_feats is not None:
                cond_feats = resample(cond_feats, mode='down') 
                cond_feats_down.append(cond_feats)
                
        #-------------------------------------------------------------------    
        # Decoder.
        if cond_feats is not None:
            cond_feats = cond_feats_down.pop()

        for name, block in self.dec.items():
            if 'block' in name:
                x = mp_cat(x, skips.pop(), t=self.concat_balance)
            x = block(x, emb, cond_feats)
            if 'up' in name and cond_feats is not None:
                cond_feats = cond_feats_down.pop()
        x = self.out_conv(x, gain=self.out_gain)
        return x

#----------------------------------------------------------------------------
# EDM2SE preconditioned U-Net model.

class EDM2SE(torch.nn.Module):
    def __init__(self,
        freq_resolution     = 256,   # Frequency resolution.
        spec_channels       = 2,     # Spectrogram channels.
        embed_channels      = 0,     # Speaker embedding channels.
        cond_feat_channels  = 2,     # Video feature channels.
        use_fp16            = True,  # Run the model at FP16 precision?
        sigma_x             = 0.402, # Expected standard deviation of clean speech.
        sigma_n             = 0.342, # Expected standard deviation of environmental noise.
        T                   = 1.0,   # Final time step.
        logvar_channels     = 128,   # Intermediate dimensionality for uncertainty estimation.
        eps                 = 1e-8,  # Small constant to prevent division by zero.
        k                   = 2.6,   # Constant for the Schroedinger bridge.
        c                   = 0.4,   # Constant for the Schroedinger bridge.
        c_s                 = 1,     # Noise prediction (1) or clean speech prediction (0)
        **unet_kwargs,               # Keyword arguments for UNet.
    ):
        super().__init__()
        self.freq_resolution = freq_resolution
        self.spec_channels = spec_channels
        self.embed_channels = embed_channels
        self.cond_feat_channels = cond_feat_channels
        self.use_fp16 = use_fp16
        self.sigma_x = sigma_x
        self.sigma_n = sigma_n
        self.T = T
        self.unet = UNet(
            freq_resolution=freq_resolution, 
            spec_channels=spec_channels, 
            input_channels=spec_channels,     
            output_channels=spec_channels, 
            embed_channels=embed_channels, 
            cond_feat_channels=cond_feat_channels,
            **unet_kwargs
            )
        self.logvar_fourier = MPFourier(logvar_channels)
        self.logvar_linear = MPConv(logvar_channels, 1, kernel=[])
        self.eps = eps
        self.k = torch.tensor(k, dtype=torch.float32)
        self.c = torch.tensor(c, dtype=torch.float32)
        self.c_s = c_s

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
        w_x = self.alpha(t) * self.sigma_bar(t) **2 / (self.sigma(self.T) ** 2 + self.eps)             
        return w_x

    def w_y(self, t):
        w_y = self.alpha_bar(t) * self.sigma(t) ** 2 / (self.sigma(self.T) ** 2 + self.eps)              
        return w_y

    def std(self, t):
        std = (self.alpha(t) * self.sigma_bar(t) * self.sigma(t)) / (self.sigma(self.T) + self.eps)
        return std

    def c_skip(self, t):
        if self.c_s == 1:
            c_skip = torch.ones_like(t)
        elif self.c_s == 0:
            c_skip = torch.zeros_like(t)
        else:
            raise ValueError(
                f"Invalid value for c_s: {self.c_s}. "
                "Expected c_s to be either 0 or 1."
            )
        return c_skip

    # For c_skip = 1
    def c_out(self, t):
        if self.c_s == 1:
            w_x = self.w_x(t)
            w_y = self.w_y(t)
            c_out = torch.sqrt((1 - w_x - w_y)**2 * self.sigma_x ** 2 + w_y ** 2 * self.sigma_n ** 2 + self.std(t) ** 2)
        elif self.c_s == 0:
            c_out = self.sigma_x   
        else:
            raise ValueError(
                f"Invalid value for c_s: {self.c_s}. "
                "Expected c_s to be either 0 or 1."
            )
        return c_out

    def c_in(self, t):
        w_x = self.w_x(t)
        w_y = self.w_y(t)
        c_in = 1 / ((w_x + w_y) ** 2 * self.sigma_x ** 2 + w_y ** 2 * self.sigma_n ** 2 + self.std(t) ** 2)
        return c_in

    def forward(self, x, t, spk_embed=None, cond_feats=None, force_fp32=False, return_logvar=False, **unet_kwargs):
        x = x.to(torch.float32)

        # Speaker embedding.
        spk_embed = None if self.embed_channels == 0 else torch.zeros([1, self.embed_channels], device=x.device) if spk_embed is None else spk_embed.to(torch.float32).reshape(-1, self.embed_channels)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        # Calculate noise standard deviation.
        t = t.to(torch.float32).reshape(-1, 1, 1, 1)
        T = self.T * torch.ones_like(t)
        sdt = self.std(t) 
        
        # Input preconditioning.
        x_in = (self.c_in(t) * x).to(dtype)
        cond_feats = (self.c_in(T) * cond_feats).to(dtype) if cond_feats is not None else None
        # cond_feats = cond_feats.to(dtype) if cond_feats is not None else None
        c_noise = sdt.flatten().log() / 4

        # Run the model.
        F_x = self.unet(x_in, c_noise, spk_embed, cond_feats, **unet_kwargs)

        # Output preconditioning.
        D_x = self.c_skip(t) * x + self.c_out(t) * F_x.to(torch.float32)

        # Estimate uncertainty if requested.
        if return_logvar:
            logvar = self.logvar_linear(self.logvar_fourier(c_noise)).reshape(-1, 1, 1, 1)
            return D_x, logvar # u(sigma) in Equation 21
        return D_x

