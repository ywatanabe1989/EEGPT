#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-08 20:44:32 (ywatanabe)"

import logging
import random
import re
import sqlite3

import mne
import mngs
import numpy as np
import torch
from scripts.data.load.get_metadata import get_metadata
from scripts.data.load.load import load_a_processed_and_segmented_eeg_file


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
            segment BLOB NOT NULL,
            metadata TEXT  -- Column to store metadata as JSON string
        )
        """
    )
    count_populated = 0
    datasets = []
    subject_ids = []
    for filepath in files_list:
        print(f"\nPopulating: {filepath}")

        dataset_id, subject_id, is_retrieved = extract_ids(filepath)

        if not is_retrieved:
            logging.warning(
                f"dataset_id or subject_id were not extracted from {filepath}"
            )
            continue

        # Makes subject_id identical across datasets
        subject_id = dataset_id + "_" + subject_id
        print(dataset_id)
        print(subject_id)

        # Loads EEG segments
        eeg_segments, metadatas = load_a_processed_and_segmented_eeg_file(
            filepath
        )

        if eeg_segments is None:
            continue

        # Normalize the EEG segments
        eeg_segments_z = mngs.gen.to_z(eeg_segments, axis=-1)
        eeg_segments_z = eeg_segments_z[
            ~np.isnan(eeg_segments).any(axis=-1).any(axis=-1)
        ]

        # Insert segments and metadata into the database
        for i_seg, (segment, metadata) in enumerate(
            zip(eeg_segments_z, metadatas)
        ):
            # fixme: segment_number may not be necessary
            segment_bytes = segment.tobytes()
            cursor.execute(
                "INSERT INTO eeg_data (dataset_id, subject_id, segment_number, segment, metadata) VALUES (?, ?, ?, ?, ?)",
                (dataset_id, subject_id, i_seg, segment_bytes, metadata),
            )

        subject_ids.append(subject_id)
        datasets.append(dataset_id)
        count_populated += 1

    conn.commit()

    # Print the summary
    cursor.execute("SELECT DISTINCT dataset_id FROM eeg_data")
    datasets = [row[0] for row in cursor.fetchall()]
    cursor.execute("SELECT DISTINCT subject_id FROM eeg_data")
    subject_ids = [row[0] for row in cursor.fetchall()]
    print(subject_ids)
    conn.close()

    n_datasets = len(np.unique(datasets))
    n_subjects = len(
        np.unique(mngs.ml.utils.merge_labels(datasets, subject_ids))
    )
    perc_populated = round(100 * (count_populated / len(files_list)), 2)
    print(
        f"\nConglaturations! SQLite databse was created from {count_populated} files ({perc_populated}%), "
        f"{n_datasets} datasets, and {n_subjects} subjects:\n{spath_db}"
    )


def extract_ids(filepath):
    pattern = r".*/ds(?P<dataset_id>[a-zA-Z0-9_-]+).*?(?:/sub-|/S)(?P<subject_id>[a-zA-Z0-9]+).*"
    match = re.search(pattern, filepath)

    try:
        dataset_id = "ds" + match.group("dataset_id")
        prefix_sub = "sub-" if "/sub-" in filepath else "S"
        subject_id = prefix_sub + match.group("subject_id")
        is_retrieved = True
    except Exception as e:
        dataset_id, subject_id = None, None
        is_retrieved = False

    return dataset_id, subject_id, is_retrieved


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
        help="[Openneuro, Dementia, Epilepsy, MentalMovementImagery, Visual or WorkingMemory]",
        # default=["epilepsy"],
        default=["motormovementimagery"],
    )
    args = parser.parse_args()

    CONFIG, sys.stdout, sys.stderr, plt, cc = mngs.gen.start(sys, plt)

    mngs.gen.fix_seeds(np=np, random=random)

    # Process directories and populate the database
    for data_cat in args.data_category:
        spath_db = mngs.io.increment_version(
            CONFIG[f"DIR_{data_cat.upper()}"] + "dbs/", "eeg_data", ".db"
        )
        if CONFIG["IS_DEBUG"]:
            spath_db = spath_db.replace("/eeg_data", "/debug_eeg_data")
        LPATHS = mngs.io.find(
            CONFIG[f"DIR_{data_cat.upper()}"] + "raw/",
            "f",
            [f"*.{ext}" for ext in CONFIG["EEG_EXTENSIONS"]],
        )
        LPATHS = np.random.permutation(LPATHS)[: CONFIG["N_LPATHS_TO_LOAD"]]
        count_datasets(LPATHS)
        populate_db(LPATHS, spath_db)

    mngs.gen.close(CONFIG)

    """
    screen -S populate
    ./scripts/data/database/populate_db.py --data_category OpenNeuro
    ./scripts/data/database/populate_db.py --data_category Dementia    
    ./scripts/data/database/populate_db.py --data_category Epilepsy
    ./scripts/data/database/populate_db.py --data_category MotorMovementImagery
    """

    ## EOF
