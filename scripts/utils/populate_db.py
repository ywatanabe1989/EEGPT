#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-01-27 14:40:20 (ywatanabe)"

import re
import sqlite3

import mne
import mngs
import numpy as np
import torch
from scripts.utils.load_eeg_files import load_an_eeg_file
from torch.utils.data import DataLoader, Dataset

CONFIG = mngs.io.load("./config/global.yaml")


def preprocess_raw(raw):
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
    return raw


# def segment_raw(raw, CONFIG):
def segment_raw(raw):
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


# def load_processed_eeg_file(filepath, CONFIG):
def load_processed_eeg_file(filepath):
    try:
        with mngs.gen.suppress_output():
            raw = load_an_eeg_file(filepath)
            if raw is not None:
                raw = preprocess_raw(raw)
                segs = segment_raw(raw)
                return segs
            else:
                return None
    except Exception as e:
        print(f"\nError:\n{filepath} was not populated.\n{e}")


# Create and populate the database
def create_and_populate_db(files_list):
    conn = sqlite3.connect(CONFIG["DB_PATH"])
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
    count = 0
    datasets = []
    subject_ids = []
    for filepath in files_list:
        print(f"\nPopulating: {filepath}")

        dataset_id, subject_id, is_retrieved = extract_ids(filepath)

        if not is_retrieved:
            continue

        eeg_segments = load_processed_eeg_file(filepath)

        if eeg_segments is None:
            continue

        eeg_segments_z = mngs.gen.to_z(eeg_segments, axis=-1)
        eeg_segments_z = eeg_segments_z[
            ~np.isnan(eeg_segments).any(axis=-1).any(axis=-1)
        ]

        for i, segment in enumerate(eeg_segments_z):
            segment_bytes = segment.tobytes()
            cursor.execute(
                "INSERT INTO eeg_data (dataset_id, subject_id, segment_number, segment) VALUES (?, ?, ?, ?)",
                (dataset_id, subject_id, i, segment_bytes),
            )

        subject_ids.append(subject_id)
        datasets.append(dataset_id)
        count += 1

    conn.commit()
    conn.close()

    n_datasets = len(np.unique(datasets))
    n_subjects = len(
        np.unique(mngs.ml.utils.merge_labels(datasets, subject_ids))
    )
    print(
        f"\nConglaturations! SQLite databse was created from {count} files, "
        f"{n_datasets} datasets, and {n_subjects} subjects:\n{CONFIG['DB_PATH']}"
    )


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


def prompt_to_user(db_path):
    response = input(f"Are you sure to populate {db_path}? (y/n) ")
    # if response.lower() == 'y':
    #     # Code to populate CONFIG['DB_PATH']
    #     pass
    # elif response.lower() == 'n':
    #     # Code to handle negative response
    #     pass
    # else:
    #     print("Invalid input. Please enter 'y' for yes or 'n' for no.")
    return response.lower()


if __name__ == "__main__":
    # Usage
    import sys

    import matplotlib.pyplot as plt
    import mngs
    from tqdm import tqdm

    CONFIG, sys.stdout, sys.stderr, plt = mngs.gen.start(sys, plt)

    y_or_n = prompt_to_user(CONFIG["DB_PATH"])

    if y_or_n == "y":
        LPATHS = np.random.permutation(
            mngs.io.load("./data/eeg_files.txt")
        )  # [
        #     :1000
        # ]
        n_lpaths_uq = len(np.unique([lpath.split("/")[2] for lpath in LPATHS]))
        print(f"\nMaximum of {n_lpaths_uq} datasets will be populated.")
        create_and_populate_db(LPATHS)

    mngs.gen.close(CONFIG)

    """
    find ./data/ -name '*.edf' > ./data/files_list.txt
    find ./data/ -name '*.bdf' >> ./data/files_list.txt
    find ./data/ -name '*.set' >> ./data/files_list.txt

    screen -S populate
    sshell
    source env/bin/activate
    python3.9 ./scripts/utils/populate_db.py

    # du ./data/ # 2.5T
    """
