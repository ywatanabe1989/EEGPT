#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-08-04 08:13:34 (ywatanabe)"

import os
import random
import mne
import torch
from mne_bids import BIDSPath, read_raw_bids
from mne.channels import make_standard_montage
from torch.utils.data import Dataset, DataLoader
import warnings




class EEGDataset(Dataset):
    def __init__(self, bids_roots, t_segment=10, fs=128):
        self.bids_roots = bids_roots if isinstance(bids_roots, list) else [bids_roots]
        self.t_segment = t_segment
        self.fs = fs
        self.segments = self.get_segments()

    def get_subjects(self, bids_root):
        subjects = []
        for subj_folder in os.listdir(bids_root):
            if "sub" in subj_folder:
                subjects.append((subj_folder, bids_root))
        return subjects

    def get_raw(self, subject, root):
        subject_id = subject.split('-')[1]  # Extract the subject id without the 'sub-' prefix
        bids_path = BIDSPath(subject=subject_id, root=root, datatype='eeg')

        try:
            all_matching_paths = bids_path.match()
            if len(all_matching_paths) > 0:
                bids_path = random.choice(all_matching_paths)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Did not find any eeg.json associated with")
                    raw = read_raw_bids(bids_path, verbose=False)

                # Resample to 128 Hz
                raw.resample(self.fs)

                # Adjust channels (keep only EEG channels and exclude bad ones)
                raw.pick(picks="eeg", exclude='bads')

                # Load standard montage (modify according to your data)
                montage = make_standard_montage('standard_1020')
                raw.set_montage(montage, match_case=False)

                print(f"Subject {subject_id} from {root} was successfully loaded.")
                return raw

            else:
                print(f"No EEG data found for subject {subject_id}.")
                return None

        except Exception as e:
            print(f"Error with subject {subject_id} in {bids_path} due to error: {str(e)}")
            return None    
    

    def get_segments(self):
        segments = []
        for bids_root in self.bids_roots:
            subjects = self.get_subjects(bids_root)
            for subject, root in subjects:
                raw = self.get_raw(subject, root)
                if raw is not None:  # Only generate segments if the raw data was loaded successfully
                    n_segments = raw.times[-1] // self.t_segment
                    segments += [(subject, root, i*self.t_segment, (i+1)*self.t_segment) for i in range(int(n_segments))]
        return segments    
    

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        subject, root, t_start, t_end = self.segments[idx]
        raw = self.get_raw(subject, root)
        data, times = raw[
            :, int(t_start * raw.info["sfreq"]) : int(t_end * raw.info["sfreq"])
        ]
        return torch.from_numpy(data), subject, (t_start, t_end)


def collate_fn(batch):
    data, subjects, time_windows = zip(*batch)
    return torch.stack(data), subjects, time_windows


if __name__ == "__main__":
    from glob import glob

    bids_roots = glob("./data/EEGPT_datasets/*")
    # bids_roots = ['/path/to/your/bids/dataset1', '/path/to/your/bids/dataset2']  # add more paths as needed
    dataset = EEGDataset(bids_roots)

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched)
        if i_batch == 5:  # stop after 5 batches for testing
            break

        
