# Copyright (c) 2026, Signal Processing Group. University of Hamburg. All rights reserved.
#
# Code adapted from https://github.com/NVlabs/edm2
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

import torch
import numpy as np

from torch_utils import misc
from einops import rearrange


class Encoder:
    def __init__(self):
        pass

    def init(self, device): # force lazy init to happen now
        pass

    def __getstate__(self):
        return self.__dict__

    def encode(self, x): # raw data => latents
        raise NotImplementedError # to be overridden by subclass

    def decode(self, x): # latents => raw data
        raise NotImplementedError # to be overridden by subclass


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """Dynamique range compression for audio signals"""
    return torch.log(torch.clamp(x, min=clip_val) * C)

#----------------------------------------------------------------------------
# Encoders/decoders for audio data.
    
class SpectrogramEncoder(Encoder):
    def __init__(
        self,
        sr = 16000,
        hop_length = 128,
        win_length = 510,
        n_fft = 510,
        spec_exp = 0.5,
        spec_factor = 0.15,
        raw_std = 0.08,
        final_std = 0.5,
        device = torch.device("cuda"),
    ):
        super().__init__()
        
        self.sr = sr
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        self.spec_exp = torch.tensor(spec_exp)
        self.spec_factor = torch.tensor(spec_factor)
        self.scale = np.float32(final_std) / np.float32(raw_std)
        self.window = torch.hann_window(self.win_length, periodic=True)
        
    def init(self, device): # force lazy init to happen now
        super().init(device)
        self.window = self.window.to(device)
        self.spec_factor = self.spec_factor.to(device)
        self.spec_exp = self.spec_exp.to(device)

    def spec_fwd(self, spec):
        spec = spec.abs()**self.spec_exp * torch.exp(1j * spec.angle())
        spec = spec * self.spec_factor
        return spec

    def spec_back(self, spec):
        spec = spec / self.spec_factor
        spec = spec.abs()**(1/self.spec_exp) * torch.exp(1j * spec.angle())
        return spec

    def encode(self, x): # raw latents => final latents
        self.init(x.device)
        x = torch.stft(x, self.n_fft, self.hop_length, self.win_length, window=self.window, return_complex=True)
        x = self.spec_fwd(x)
        x = rearrange(torch.view_as_real(x.squeeze(1)), 'b f t c -> b c f t')
        x = x * misc.const_like(x, self.scale).reshape(1, -1, 1, 1)
        return x

    def decode(self, x): # mels => raw samples
        self.init(x.device)
        x = x / misc.const_like(x, self.scale).reshape(1, -1, 1, 1)
        x = rearrange(x, 'b c f t -> b f t c').contiguous()
        x = torch.view_as_complex(x)
        x = self.spec_back(x)
        x = torch.istft(x, self.n_fft, self.hop_length, self.win_length, window=self.window)
        return x

