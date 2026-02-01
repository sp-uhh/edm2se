# Copyright (c) 2026, Signal Processing Group. University of Hamburg. All rights reserved.
#
# Code adapted from https://github.com/NVlabs/edm2
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Dataset for audio restoration tasks."""

import torch
import torchaudio
import numpy as np
from glob import glob
from os.path import join


#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        path,                # Path to the directory.
        subset = 'train',    # Subset of the dataset to load.
        sr = 16000,          # Sample rate for audio.
        num_frames = 256,    # Number of frames in the spectrogram.
        hop_length = 128,    # Hop length for the spectrogram.
        normalize = True,    # Normalize audio to [-1, 1]?
        **super_kwargs,      # Additional arguments for the Dataset base class.
    ):
        self.path = path
        self.subset = subset
        self.sr = sr
        self.num_frames = num_frames
        self.hop_length = hop_length
        self.normalize = normalize

        # Audio paths.
        self.clean_paths = sorted(
            glob(join(path, subset, 'clean', '**', '*.wav'), recursive=True)
        )

        self.noisy_paths = sorted(
            glob(join(path, subset, 'noisy', '**', '*.wav'), recursive=True)
        )
    
    def __len__(self):
        return len(self.clean_paths)

    def __getitem__(self, idx):
        # Load audio.
        x, sr_x = torchaudio.load(self.clean_paths[idx])
        y, sr_y = torchaudio.load(self.noisy_paths[idx])

        # Check sample rate and shape.
        assert sr_x == self.sr, (
            f"Sample rate mismatch in {self.clean_paths[idx]}: "
            f"{sr_x} != {self.sr}"
        )

        assert sr_y == self.sr, (
            f"Sample rate mismatch in {self.noisy_paths[idx]}: "
            f"{sr_y} != {self.sr}"
        )

        # Ensure equal time dimensions by truncating the longer signal.
        if x.shape != y.shape:
            if x.shape[1] > y.shape[1]:
                x = x[:, :y.shape[1]]
            else:
                y = y[:, :x.shape[1]]

        # Take only the first channel.
        x = x[0]
        y = y[0]

        # Normalize audio with respect to the maximum value in the noisy audio.
        if self.normalize:
            normfac = y.abs().max()
            x = x / normfac
            y = y / normfac

        # Cut audio to fixed length.
        target_len = (self.num_frames - 1) * self.hop_length
        current_len = x.size(-1)
        pad = max(target_len - current_len, 0)
        if pad == 0:
            # Extract random part of the audio file.
            start = int(np.random.uniform(0, current_len-target_len))
            x = x[start:start + target_len]
            y = y[start:start + target_len]
        else:
            # Repeat the signal to pad to the desired length
            repeat_times = (target_len // current_len) + 1
            x = x.repeat(repeat_times)[:target_len]
            y = y.repeat(repeat_times)[:target_len]

        # Return the item.
        item = {
            "clean_audio": x,
            "noisy_audio": y,
        }

        return item

    def collate_fn(self, batch):
        # List to hold the batched data for each item in the dictionary
        collated_batch = {}
        
        # Assuming each item in batch is a dictionary with the same keys
        if len(batch) == 0:
            return collated_batch

        first_item = batch[0]
        for key in first_item.keys():
            # Gather all elements under the key across the entire batch
            data_list = [item[key] for item in batch]

            # Stack or pad/stack if necessary
            if isinstance(data_list[0], torch.Tensor):
                collated_batch[key] = torch.stack(data_list)
            else:
                # Additional handling if the type is not a tensor
                if data_list[0] is not None:
                    collated_batch[key] = torch.tensor(np.array(data_list))
                else:
                    collated_batch[key] = None

        return collated_batch



