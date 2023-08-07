#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-08-07 07:49:58 (ywatanabe)"

from glob import glob
from bids import BIDSLayout

bids_roots = glob("./data/EEGPT_datasets/ds*")
for bids_root in bids_roots:
    try:
        layout = BIDSLayout(bids_root, validate=True)
        print(f"BIDS dataset at {bids_root} passed the validation.")
    except Exception as e:
        print(f"BIDS dataset at {bids_root} failed the validation with error {e}")
