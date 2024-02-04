#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-04 16:13:08 (ywatanabe)"

import random

import mngs
import numpy as np
from scripts.utils.preprocess import preprocess_raw

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


def segment_raw(raw):
    try:
        # Convert to DataFrame and transpose
        df = raw.to_data_frame().set_index("time").T

        # Segmentation
        segs = mngs.dsp.crop(
            np.array(df),
            int(CONFIG["WINDOW_SIZE_SEC"] * CONFIG["TGT_FS"]),  # 256
            overlap_factor=CONFIG["OVERLAP_FACTOR"],
        )

        # Limit the number of segments to reduce sampling bias
        segs = segs[
            np.random.permutation(np.arange(len(segs)))[
                : CONFIG["MAX_N_WINDOWS_PER_SUB"]
            ]
        ]
        return segs

    except Exception as e:
        print(f"\nERROR:\nNot segmented.\n{e}")
        return None


def load_a_processed_and_segmented_eeg_file(filepath):
    def return_if_not_none(fn, arg):
        if arg is None:
            return None
        else:
            return fn(arg)

    raw = return_if_not_none(_load_an_eeg_file, filepath)
    preprocessed_raw = return_if_not_none(preprocess_raw, raw)
    segs = return_if_not_none(segment_raw, preprocessed_raw)
    return segs


# def _load_processed_eeg_files(lpaths):
#     return np.vstack([load_a_processed_eeg_file(lpath) for lpath in LPATHS])


if __name__ == "__main__":
    LPATHS = find_eeg_files("dementia")
    # loaded = load_eeg_files(LPATHS, n_loads=100)
    segs = load_a_processed_and_segmented_eeg_file(LPATHS[0])
    # print(segs)
