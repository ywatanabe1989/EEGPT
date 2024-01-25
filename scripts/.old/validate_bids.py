#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-08-08 10:42:28 (ywatanabe)"

from glob import glob
from bids import BIDSLayout

bids_roots = glob("./data/EEGPT_datasets/ds*")
for bids_root in bids_roots:
    try:
        layout = BIDSLayout(bids_root, validate=True)
        print(f"BIDS dataset at {bids_root} passed the validation.")
    except Exception as e:
        print(f"BIDS dataset at {bids_root} failed the validation with error {e}")

# The number of subjects in BIDS data
bids_roots = glob("./data/EEGPT_datasets/ds*")
for bids_root in bids_roots:
    layout = BIDSLayout(bids_root)
    subjects = layout.get_subjects()
    num_subjects = len(subjects)
    print(f"The number of subjects in the BIDS dataset is: {num_subjects}")
# np.sum([31, 11, 122, 88]) # 252
        
