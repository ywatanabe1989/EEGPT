#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-01-26 12:51:26 (ywatanabe)"

import re
import sqlite3

import mne
import mngs
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

CONFIG = mngs.io.load("./config/global.yaml")


def load_eeg_file(filepath):
    try:
        with mngs.gen.suppress_output():
            # Loads
            try:
                raw = mne.io.read_raw(filepath, preload=True, verbose=False)
            except Exception as e:
                print(e)
                return None

            # Converts channel names to uppercase
            raw.rename_channels(lambda x: x.upper())

            # Maps channels
            try:
                raw.rename_channels(CONFIG["CHANNEL_MAPPING"])
            except Exception as e:
                print("Channel mapping was not conducted:", e)

            # Sets montages
            raw.set_montage(
                mngs.dsp.to_dig_montage(CONFIG["TGT_MONTAGE"]),
                on_missing="ignore",
                match_case=False,
            )

            # Picks predefined channels
            raw = raw.pick(CONFIG["TGT_MONTAGE"])

            # Re-reference
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
                )
            except Exception as e:
                print(e)
                return None

            # Notch filtering at 50 and 60 Hz
            raw = raw.notch_filter(freqs=[50, 60], n_jobs=1)

            # Interpolate bad channels if any
            raw = raw.interpolate_bads(verbose=False)

            # Downsampling to CONFIG["TGT_FS"]
            raw.resample(CONFIG["TGT_FS"], n_jobs=1)

            # Convert to DataFrame and transpose
            df = raw.to_data_frame().set_index("time").T

            # Segmentation
            segs = mngs.dsp.crop(
                np.array(df),
                int(CONFIG["WINDOW_SIZE_SEC"] * CONFIG["TGT_FS"]),  # 256
                overlap_factor=CONFIG["OVERLAP_FACTOR"],
            )

            # Extract segments
            segs = segs[
                np.random.permutation(np.arange(len(segs)))[
                    : CONFIG["MAX_N_WINDOWS_PER_SUB"]
                ]
            ]

            return segs

    except Exception as e:
        print(f"\nError:\n{filepath} was not populated.\n{e}")


# Create and populate the database
def create_and_populate_db(files_list, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Drop the existing table if it exists
    cursor.execute("DROP TABLE IF EXISTS eeg_data")

    # Recreate the table with the correct schema
    cursor.execute(
        """
        CREATE TABLE eeg_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id TEXT NOT NULL,        
            subject_id TEXT NOT NULL,
            segment_number INTEGER NOT NULL,
            segment BLOB NOT NULL
        )
        """
    )
    for filepath in files_list:
        print(f"\nPopulating: {filepath}")

        dataset_id, subject_id, is_retrieved = extract_ids(filepath)

        if not is_retrieved:
            continue

        eeg_segments = load_eeg_file(filepath)
        if eeg_segments is None:
            continue
        eeg_segments_z = mngs.gen.to_z(eeg_segments, axis=-1)

        for i, segment in enumerate(eeg_segments_z):
            segment_bytes = segment.tobytes()
            cursor.execute(
                "INSERT INTO eeg_data (dataset_id, subject_id, segment_number, segment) VALUES (?, ?, ?, ?)",
                (dataset_id, subject_id, i, segment_bytes),
            )
    conn.commit()
    conn.close()
    print(f"\nConglaturations! SQLite databse was created:\n{db_path}")


def extract_ids(filepath):
    pattern = r".*/ds(?P<dataset_id>\d+).*?/sub-(?P<subject_id>[a-zA-Z0-9]+).*"
    match = re.search(pattern, filepath)
    is_retrieved = False
    try:
        dataset_id = "ds" + match.group("dataset_id")
        subject_id = "sub-" + match.group("subject_id")
        is_retrieved = True
    except Exception as e:
        dataset_id, subject_id = None, None
        pass

    return dataset_id, subject_id, is_retrieved


if __name__ == "__main__":
    # Usage
    import os
    import sys
    from datetime import datetime

    import matplotlib.pyplot as plt
    import mngs

    CONFIG, sys.stdout, sys.stderr, plt = mngs.gen.start(sys, plt)

    # if os.path.exists(CONFIG["DB_PATH"]):
    #     os.remove(CONFIG["DB_PATH"])

    EXTS = [".edf", ".bdf", ".set"]  # ".fdt",
    LPATHS = np.hstack(
        [mngs.gen.find(CONFIG["DATA_DIR"], "*" + ext) for ext in EXTS]
    )

    create_and_populate_db(
        LPATHS, "./data/eeg_data_v3.db"
    )  # CONFIG["DB_PATH"])

    mngs.gen.close(CONFIG)

    """
    python ./scripts/utils/populate_db.py
    """
