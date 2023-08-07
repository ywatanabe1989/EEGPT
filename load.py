#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-08-07 12:35:31 (ywatanabe)"

import os
import random
import mne
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from joblib import Parallel, delayed
from bids import BIDSLayout
from glob import glob

import warnings

warnings.filterwarnings("ignore")

class BidsDataset(Dataset):
    def __init__(self, bids_roots, t_segment=10, sampling_rate=128):
        self.bids_roots = bids_roots
        self.t_segment = t_segment
        self.sampling_rate = sampling_rate
        montage = mne.channels.make_standard_montage("standard_1020")
        montage.rename_channels(lambda x: x.upper())  # Convert channel names to uppercase
        self.montage = montage


        self.common_channels = [
            "FP1",
            "FP2",
            "F7",
            "F3",
            "FZ",
            "F4",
            "F8",
            "T7",
            "C3",
            "CZ",
            "C4",
            "T8",
            "P7",
            "P3",
            "PZ",
            "P4",
            "P8",
            "O1",
            "O2",
        ]

        self.layouts = [BIDSLayout(root) for root in bids_roots]

        self.files = np.hstack([layout.get(extension=[".edf", ".bdf", ".set"]) for layout in self.layouts])
        self.files = [str(f.path) for f in self.files]  # Convert file paths to strings        
                    

    def __len__(self):
        return len(self.files)  # Return the length of self.files

    def __getitem__(self, index):
        f = self.files[index]  # Get the file at the given index        
        channel_mapping = {
            "T3": "T7",
            "T4": "T8",
            "T5": "P7",
            "T6": "P8",
        }
        

        try:
            raw = mne.io.read_raw(f, preload=True, verbose=False)
            raw.rename_channels(lambda x: x.upper())  # Convert channel names to uppercase
            channel_mapping = {old: new for old, new in channel_mapping.items() if old in raw.ch_names}
            raw.rename_channels(channel_mapping)
            raw.set_montage(self.montage, on_missing="ignore")
            # Pick the channels that are present in both the raw data and the montage
            raw = raw.pick(self.common_channels)
            raw.resample(self.sampling_rate)        
            
            # Randomly select a time segment
            total_time = raw.times[-1]
            start_time = random.uniform(0, total_time - self.t_segment)
            stop_time = start_time + self.t_segment

            # Extract EEG data for the time segment
            segment, times = raw[
                :,
                int(start_time * raw.info["sfreq"]) : int(
                    stop_time * raw.info["sfreq"]
                ),
            ]

            # Normalize the data
            segment = (segment - np.mean(segment)) / np.std(segment)

            return torch.from_numpy(np.array(segment))

        except Exception as e:
            print(f"Error processing file {f}: {e}")
            return None


# List of BIDS dataset roots
bids_roots = glob("./data/EEGPT_datasets/ds*")
bids_roots = np.array(bids_roots)[[0,1,2,3]] #1,2,3

# Create BidsDataset
bids_dataset = BidsDataset(bids_roots)

# Create DataLoader
batch_size = 64


# Load a batch of data
for epoch in range(10):
    data_loader = DataLoader(
        bids_dataset, batch_size=batch_size, shuffle=True, num_workers=32
    )

    for i_batch, batch in enumerate(data_loader):
        print(epoch, i_batch)
batch = next(iter(data_loader))
print(batch.shape)

