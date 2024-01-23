#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-10-19 11:10:49 (ywatanabe)"

import os
import numpy as np
import torch
import random
import mne
from torch.utils.data import Dataset, DataLoader
from bids import BIDSLayout
from glob import glob
import warnings
import concurrent.futures

warnings.filterwarnings("ignore")

def load_and_preprocess_data(file_path, t_segment, sampling_rate, common_channels, montage):
    channel_mapping = {
        "T3": "T7",
        "T4": "T8",
        "T5": "P7",
        "T6": "P8",
    }

    try:
        raw = mne.io.read_raw(file_path, preload=True, verbose=False)
        raw.rename_channels(lambda x: x.upper())  # Convert channel names to uppercase
        channel_mapping = {
            old: new for old, new in channel_mapping.items() if old in raw.ch_names
        }
        raw.rename_channels(channel_mapping)
        raw.set_montage(montage, on_missing="ignore")
        raw = raw.pick(common_channels)
        raw.resample(sampling_rate)

        # Randomly select a time segment
        total_time = raw.times[-1]
        start_time = random.uniform(0, total_time - t_segment)
        stop_time = start_time + t_segment

        # Extract EEG data for the time segment
        segment, times = raw[
            :,
            int(start_time * raw.info["sfreq"]): int(stop_time * raw.info["sfreq"]),
        ]

        # Common average
        segment -= segment.mean(axis=0, keepdims=True)

        # Normalize the data
        segment = (segment - np.mean(segment)) / np.std(segment)

        return segment

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def async_load_and_preprocess(file_list):
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        segments = list(executor.map(lambda f: load_and_preprocess_data(*f), file_list))
    return segments    
# def async_load_and_preprocess(file_list):
#     with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
#         segments = list(executor.map(lambda f: load_and_preprocess_data(*f), file_list))
#     return segments

class BidsDataset(Dataset):
    def __init__(self, bids_roots, t_segment=10, sampling_rate=128, cache_size=1000):
        self.bids_roots = bids_roots
        self.t_segment = t_segment
        self.sampling_rate = sampling_rate

        montage = mne.channels.make_standard_montage("standard_1020")
        montage.rename_channels(lambda x: x.upper())  # Convert channel names to uppercase
        self.montage = montage

        self.common_channels = [
            "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T7", "C3", 
            "CZ", "C4", "T8", "P7", "P3", "PZ", "P4", "P8", "O1", "O2"
        ]

        self.layouts = [BIDSLayout(root) for root in bids_roots]
        self.files = np.hstack([layout.get(extension=[".edf", ".bdf", ".set"]) for layout in self.layouts])
        self.files = [str(f.path) for f in self.files]  # Convert file paths to strings

        self.cache = {}
        self.cache_size = cache_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        if index in self.cache:
            data = self.cache.pop(index)
            self.cache[index] = data
        else:
            file_path = self.files[index]
            data_list = async_load_and_preprocess([(file_path, self.t_segment, self.sampling_rate, self.common_channels, self.montage) for _ in range(10)])  # preload next 10 segments
            data = data_list[0]
            for i, d in enumerate(data_list[1:], start=1):
                if len(self.cache) >= self.cache_size:
                    self.cache.pop(next(iter(self.cache)))  # Remove the oldest cached item
                self.cache[index+i] = d

        return torch.from_numpy(data).float()

if __name__ == "__main__":
    # List of BIDS dataset roots
    bids_roots = glob("./data/EEGPT_datasets/ds*")
    bids_roots = np.array(bids_roots)[[0, 1, 2, 3]]

    # Create BidsDataset
    bids_dataset = BidsDataset(bids_roots)

    # Create DataLoader
    batch_size = 64
    for epoch in range(10):
        data_loader = DataLoader(bids_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
        for i_batch, batch in enumerate(data_loader):
            print(epoch, i_batch)

    batch = next(iter(data_loader))
    print(batch.shape)
