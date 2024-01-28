#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-01-29 09:24:37 (ywatanabe)"

import re
import sqlite3

import mne
import mngs
import numpy as np
import torch
from scripts.utils.load_eeg_files import load_processed_eeg_file
from torch.utils.data import DataLoader, Dataset


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
        f"{n_datasets} datasets, and {n_subjects} subjects:\n{db_path}"
    )


def extract_ids(filepath):
    # pattern = r".*/ds(?P<dataset_id>\d+).*?/sub-(?P<subject_id>[a-zA-Z0-9]+).*"
    # pattern = r".*/ds(?P<dataset_id>[a-zA-Z0-9]+).*?/sub-(?P<subject_id>[a-zA-Z0-9]+).*"  # [REVISED]
    pattern = r".*/ds(?P<dataset_id>[a-zA-Z0-9_]+).*?/sub-(?P<subject_id>[a-zA-Z0-9]+).*"  # [REVISED]
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
    return response.lower()


def count_n_datasets(lpaths):
    n_datasets_uq = len(np.unique([lpath.split("/")[3] for lpath in lpaths]))
    print(f"\nMaximum of {n_datasets_uq} datasets will be populated.")
    return n_datasets_uq


if __name__ == "__main__":
    # Usage
    import sys

    import matplotlib.pyplot as plt
    import mngs
    from tqdm import tqdm

    CONFIG, sys.stdout, sys.stderr, plt = mngs.gen.start(sys, plt)

    # Check user intension
    y_or_n_INTERNAL = prompt_to_user(CONFIG["DB_PATH_INTERNAL"])
    y_or_n_OPENNEURO = prompt_to_user(CONFIG["DB_PATH_OPENNEURO"])
    _ = mngs.gen.run_shellscript("./scripts/utils/find_eeg_files.sh")

    # Populate internal datasets
    if y_or_n_INTERNAL == "y":
        _ = mngs.gen.run_shellcommand("rm", CONFIG["DB_PATH_INTERNAL"])

        # fixme: rm CONFIG["DB_PATH_INTERNAL"]
        LPATHS = np.random.permutation(
            mngs.io.load("./data/Internal/eeg_files.txt")
        )[: CONFIG["N_LPATHS_TO_LOAD"]]
        count_n_datasets(LPATHS)
        create_and_populate_db(LPATHS, CONFIG["DB_PATH_OPENNEURO"])

    # Populate OpenNeuro datasets
    if y_or_n_OPENNEURO == "y":
        _ = mngs.gen.run_shellcommand("rm", CONFIG["DB_PATH_OPENNEURO"])

        LPATHS = np.random.permutation(
            mngs.io.load("./data/OpenNeuro/eeg_files.txt")
        )[: CONFIG["N_LPATHS_TO_LOAD"]]
        count_n_datasets(LPATHS)
        create_and_populate_db(LPATHS, CONFIG["DB_PATH_OPENNEURO"])

    mngs.gen.close(CONFIG)

    """
    screen -S populate
    sshell
    source env/bin/activate
    python3.9 ./scripts/utils/populate_db.py
    """
