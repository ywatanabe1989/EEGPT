#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-06 09:11:10 (ywatanabe)"

import random

import mngs
import numpy as np
from scripts.data.preprocessing.associate_EEG_segments_and_metadata import (
    associate_EEG_segments_and_metadata,
)
from scripts.data.preprocessing.get_metadata import get_metadata
from scripts.data.preprocessing.preprocess import (
    preprocess_raw,
    segment_processed_raw,
)

CONFIG = mngs.io.load_configs(show=False)

# Functions
def find_eeg_files(category):
    EEG_EXPS = [f"*.{ext}" for ext in CONFIG["EEG_EXTENSIONS"]]
    LPATHS = mngs.gen.find(
        CONFIG[f"DIR_{category.upper()}"] + "raw/", "f", EEG_EXPS
    )
    return LPATHS


def _load_an_eeg_file(lpath):
    try:
        with mngs.gen.suppress_output():
            raw = mngs.io.load(lpath, verbose=False)
        return raw
    except Exception as e:
        print(f"\nERROR: {lpath} was not loaded: {e}")
        return None


# def _load_eeg_files(lpaths, n_loads=None):
#     print(f"\nLoading from {n_loads} of {len(lpaths)} files")

#     # Control how many files to load for debugging
#     if n_loads is not None:
#         lpaths = np.random.permutation(lpaths)[:n_loads]

#     loaded_raws = []

#     for lpath in lpaths:
#         loaded_raw = _load_an_eeg_file(lpath)

#         if loaded_raw is not None:
#             loaded_raws.append(loaded_raw)

#     print(f"\nLoaded: {len(loaded_raws) / n_loads * 100}% of {n_loads} data")
#     return loaded_raws


def load_a_processed_and_segmented_eeg_file(lpath_eeg):
    """
    LPATHS = find_eeg_files("dementia")
    segs, meta = load_a_processed_and_segmented_eeg_file(LPATHS[0])
    """

    def return_if_not_none(fn, arg):
        if arg is None:
            return None
        else:
            return fn(arg)

    raw = return_if_not_none(_load_an_eeg_file, lpath_eeg)

    metadata = get_metadata(lpath_eeg)

    # Process raw
    preprocessed_raw = return_if_not_none(preprocess_raw, raw)
    eeg_segs = return_if_not_none(segment_processed_raw, preprocessed_raw)

    # Process metadata
    segmented_metadata = associate_EEG_segments_and_metadata(
        metadata, eeg_segs
    )

    return eeg_segs, segmented_metadata


if __name__ == "__main__":
    LPATHS = find_eeg_files("dementia")
    segs, meta = load_a_processed_and_segmented_eeg_file(LPATHS[0])
