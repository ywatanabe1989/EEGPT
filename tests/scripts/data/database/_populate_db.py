#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-05 08:30:03 (ywatanabe)"

import random
import re
import sqlite3

import mne
import mngs
import numpy as np
import torch
from scripts.utils.get_metadata import get_metadata
from scripts.utils.load_eeg_files import (
    load_a_processed_and_segmented_eeg_file,
)


def populate_db(files_list, spath_db):
    conn = sqlite3.connect(spath_db)
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
            metadata TEXT  -- Column to store metadata as JSON string
        )
        """
        # please include metadata column
    )
    count = 0
    datasets = []
    subject_ids = []
    for filepath in files_list:
        print(f"\nPopulating: {filepath}")

        dataset_id, subject_id, is_retrieved = extract_ids(filepath)
        if not is_retrieved:
            continue

        # Fetches metadata for the current file
        metadata = get_metadata(filepath)

        # Makes subject_id identical across datasets
        subject_id = dataset_id + subject_id

        # Loads EEG segments
        # Now metadata is retrieved from lpath. However, these metadata targets the eegdata of lpath as a whole. So, splitting labels logic is also required.
        eeg_segments = load_a_processed_and_segmented_eeg_file(filepath)
        if eeg_segments is None:
            continue

        # Normalize the EEG segments
        eeg_segments_z = mngs.gen.to_z(eeg_segments, axis=-1)
        eeg_segments_z = eeg_segments_z[
            ~np.isnan(eeg_segments).any(axis=-1).any(axis=-1)
        ]

        # Insert segments and metadata into the database
        for i, segment in enumerate(eeg_segments_z):
            segment_bytes = segment.tobytes()
            cursor.execute(
                "INSERT INTO eeg_data (dataset_id, subject_id, segment_number, segment, metadata) VALUES (?, ?, ?, ?, ?)"(
                    dataset_id, subject_id, i, segment_bytes, metadata
                ),
            )

        subject_ids.append(subject_id)
        datasets.append(dataset_id)
        count += 1

    conn.commit()

    # Determine which subjects have more segments than the maximum allowed
    cursor.execute(
        "SELECT subject_id, COUNT(*) as segment_count FROM eeg_data GROUP BY subject_id"
    )
    subjects_to_trim = [
        row
        for row in cursor.fetchall()
        if row[1] > CONFIG["MAX_N_WINDOWS_PER_SUB"]
    ]

    # Randomly delete excess segments for subjects to reduce sampling biases
    for subject_id, segment_count in subjects_to_trim:
        # Calculate the number of segments to delete
        num_to_delete = segment_count - CONFIG["MAX_N_WINDOWS_PER_SUB"]
        # Get the segment IDs for this subject
        cursor.execute(
            "SELECT id FROM eeg_data WHERE subject_id = ?", (subject_id,)
        )
        segment_ids = [row[0] for row in cursor.fetchall()]
        # Randomly select segment IDs to delete
        ids_to_delete = random.sample(segment_ids, num_to_delete)
        # Delete the selected segments
        cursor.executemany(
            "DELETE FROM eeg_data WHERE id = ?",
            [(id,) for id in ids_to_delete],
        )

    conn.commit()

    # Print the summary
    cursor.execute("SELECT DISTINCT dataset_id FROM eeg_data")
    datasets = [row[0] for row in cursor.fetchall()]
    cursor.execute("SELECT DISTINCT subject_id FROM eeg_data")
    subject_ids = [row[0] for row in cursor.fetchall()]
    n_datasets = len(datasets)
    n_subjects = len(subject_ids)
    print(
        f"\nCongratulations! SQLite database was created from {len(files_list)} files, "
        f"{n_datasets} datasets, and {n_subjects} subjects:\n{spath_db}"
    )
    conn.close()

    # n_datasets = len(np.unique(datasets))
    # n_subjects = len(
    #     np.unique(mngs.ml.utils.merge_labels(datasets, subject_ids))
    # )
    # print(
    #     f"\nConglaturations! SQLite databse was created from {count} files, "
    #     f"{n_datasets} datasets, and {n_subjects} subjects:\n{spath_db}"
    # )


def extract_ids(filepath):
    pattern = r".*/ds(?P<dataset_id>[a-zA-Z0-9_]+).*?/sub-(?P<subject_id>[a-zA-Z0-9]+).*"
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


def prompt_to_user(spath_db):
    response = input(f"Are you sure to populate {spath_db}? (y/n) ")
    return response.lower()


def count_datasets(lpaths):
    n_datasets_uq = len(np.unique([lpath.split("/")[4] for lpath in lpaths]))
    print(f"\nMaximum of {n_datasets_uq} datasets will be populated.")
    return n_datasets_uq


if __name__ == "__main__":
    # Usage
    import argparse
    import sys

    import matplotlib.pyplot as plt
    import mngs
    from tqdm import tqdm

    parser = argparse.ArgumentParser(
        description="Populate an SQLite database with EEG data."
    )
    parser.add_argument(
        "--data_category",
        nargs="+",
        help="[openneuro, dementia, epilepsy, mmi, vis or wm]",
        default=["openneuro"],
    )
    args = parser.parse_args()

    CONFIG, sys.stdout, sys.stderr, plt, cc = mngs.gen.start(sys, plt)

    mngs.gen.fix_seeds(np=np, random=random)

    # Process directories and populate the database
    for data_cat in args.data_category:
        """
        data_cat = args.data_category[0]
        """
        spath_db = mngs.io.increment_version(
            CONFIG[f"DIR_{data_cat.upper()}"] + "dbs/", "eeg_data", ".db"
        )
        ldir_raw = CONFIG[f"DIR_{data_cat.upper()}"] + "raw/"

        LPATHS = mngs.io.find(
            ldir_raw, "f", [f"*.{ext}" for ext in CONFIG["EEG_EXTENSIONS"]]
        )
        LPATHS = np.random.permutation(LPATHS)[: CONFIG["N_LPATHS_TO_LOAD"]]
        count_datasets(LPATHS)
        populate_db(LPATHS, spath_db)

    mngs.gen.close(CONFIG)

    """
    screen -S populate
    ./scripts/utils/populate_db.py 
    """

    ## EOF
