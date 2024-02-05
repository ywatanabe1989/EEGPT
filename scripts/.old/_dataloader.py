#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-08-07 09:00:54 (ywatanabe)"

from glob import glob

import mne
import mngs
import numpy as np
import torch
from mne.io import read_raw_edf
from torch.utils.data import DataLoader, Dataset
import random

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

class EEGDataset(Dataset):
    def __init__(self, file_paths, t_segment=10, sampling_rate=128):
        self.file_paths = file_paths
        self.t_segment = t_segment
        self.sampling_rate = sampling_rate
        self.montage = mne.channels.make_standard_montage("standard_1020")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        try:
            raw = mne.io.read_raw_edf(self.file_paths[index], preload=True, verbose=False)
            raw.resample(self.sampling_rate)

            # Interpolate data to match the 10-20 system
            raw = raw.pick(picks="eeg")
            raw.set_montage(self.montage, on_missing="ignore", verbose=False)

            # Map the 10-20 montage to the data
            layout = mne.channels.make_eeg_layout(raw.info)
            raw = mne.channels.layout_to_channels(raw, layout)

            # Randomly select a time segment
            total_time = raw.times[-1]
            start_time = random.uniform(0, total_time - self.t_segment)
            stop_time = start_time + self.t_segment

            # Extract EEG data for the time segment
            segment, times = raw[
                :, int(start_time * raw.info["sfreq"]) : int(stop_time * raw.info["sfreq"])
            ]

            return np.array(segment)
        except Exception as e:
            print(f"Error processing file {self.file_paths[index]}: {e}")
            return None


# bids_roots = glob("./data/EEGPT_datasets/ds*")
edf_file_paths = mngs.io.load("./res/edf_files.npy")
# Assuming `file_paths` is a list of paths to your edf files
eeg_dataset = EEGDataset(edf_file_paths)

# Create DataLoader
batch_size = 64
data_loader = DataLoader(
    eeg_dataset, batch_size=batch_size, shuffle=True, num_workers=32
)

ii = random.randint(0, len(data_loader))
data = data_loader.dataset[ii]
print(data)

batch = next(iter(data_loader))


