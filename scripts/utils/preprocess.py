#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-04 16:24:12 (ywatanabe)"


import random
import warnings

import mne
import mngs
import numpy as np
import pandas as pd
from scipy.spatial import distance
from scripts.utils.handle_channels import handle_channels

CONFIG = mngs.io.load_configs(show=False)


def take_bipolar_rereference(raw):
    try:
        raw = mne.set_bipolar_reference(
            raw,
            anode=[
                bi.split("-")[0]
                for bi in CONFIG["TGT_TRANVERSE_BIPOLAR_MONTAGE"]
            ],
            cathode=[
                bi.split("-")[1]
                for bi in CONFIG["TGT_TRANVERSE_BIPOLAR_MONTAGE"]
            ],
            ch_name=CONFIG["TGT_TRANVERSE_BIPOLAR_MONTAGE"],
            drop_refs=True,
            verbose=False,
        )
        return raw

    except Exception as e:
        print(e)
        return None


def preprocess_raw(raw):
    try:
        # Handle channels including mapping and extraction
        raw = handle_channels(raw)

        # Take bipolar montage among 19 channels
        raw = take_bipolar_rereference(raw)

        # Notch filtering at 50 and 60 Hz
        raw = raw.notch_filter(freqs=[50, 60], n_jobs=1, verbose=False)

        # Interpolate bad channels if any
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", RuntimeWarning
            )  # for the condition when no bad channels are detected

            raw = raw.interpolate_bads(verbose=False)

        # Downsampling to CONFIG["TGT_FS"]
        raw.resample(CONFIG["TGT_FS"], n_jobs=1)

        return raw

    except Exception as e:
        print(f"\nERRROR:\nNot preprocessed.\n{e}")
        return None
