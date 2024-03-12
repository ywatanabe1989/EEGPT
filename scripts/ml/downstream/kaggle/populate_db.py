#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-03-09 19:29:20 (ywatanabe)"

"""Populates a Kaggle dataset"""

import json
import logging
import sqlite3
from glob import glob

import mne
import mngs
import numpy as np
import pandas as pd
from scripts.data.load.preprocess import preprocess_raw, segment_processed_raw
from tqdm import tqdm

CONFIG = mngs.io.load_configs()

# Functions
def kaggle_populate_db(files_list, spath_db):
    conn = sqlite3.connect(spath_db)
    cursor = conn.cursor()

    # Drop the existing table if it exists
    cursor.execute("DROP TABLE IF EXISTS eeg_data")

    # Recreate the table with the correct schema
    cursor.execute(
        """
        CREATE TABLE eeg_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_id TEXT NOT NULL,
            segment_number INTEGER NOT NULL,
            segment BLOB NOT NULL,
            eeg_id TEXT NOT NULL,
            votes BLOB NOT NULL
        )
        """
    )

    count_populated = 0
    subject_ids = []
    for filepath in tqdm(files_list):
        print(f"\nPopulating: {filepath}")

        # Loads EEG segments
        train_or_test = "train" if "train_eegs" in filepath else "test"
        eeg_segments, metadatas = load_kaggle(filepath, train_or_test)

        if eeg_segments is None:
            logging.warning(f"{filepath} was not loaded properly. Skipping.")
            continue

        subject_id = metadatas.subject_id.unique()[0]
        del metadatas["subject_id"]
        metadatas = [md for _, md in metadatas.iterrows()]  # to list

        if eeg_segments is None:
            continue

        # Normalize the EEG segments
        eeg_segments_z = mngs.gen.to_z(eeg_segments, axis=-1)
        eeg_segments_z = eeg_segments_z[
            ~np.isnan(eeg_segments).any(axis=-1).any(axis=-1)
        ]

        votes_columns = [
            "seizure_vote",
            "lpd_vote",
            "gpd_vote",
            "lrda_vote",
            "grda_vote",
            "other_vote",
        ]

        # Insert segments and metadata into the database
        for i_seg, (segment, metadata) in enumerate(
            zip(eeg_segments_z, metadatas)
        ):
            eeg_id = metadata["eeg_id"]

            votes_bytes = np.array(
                metadata[votes_columns], dtype=np.float64
            ).tobytes()

            segment_bytes = segment.tobytes()

            cursor.execute(
                "INSERT INTO eeg_data (subject_id, segment_number, segment, eeg_id, votes) VALUES (?, ?, ?, ?, ?)",
                (subject_id, i_seg, segment_bytes, eeg_id, votes_bytes),
            )

        subject_ids.append(subject_id)
        count_populated += 1

    conn.commit()

    # Print the summary
    cursor.execute("SELECT DISTINCT subject_id FROM eeg_data")
    subject_ids = [row[0] for row in cursor.fetchall()]
    print(f"\nUnique Subjects: {subject_ids}")
    conn.close()

    n_subjects = len(np.unique(subject_ids))
    perc_populated = round(100 * (count_populated / len(files_list)), 2)
    print(
        f"\nConglaturations! SQLite databse was created from {count_populated} files ({perc_populated}%), "
        f"{n_subjects} subjects:\n{spath_db}"
    )


def load_kaggle(lpath, train_or_test):
    # EEG signals
    data = mngs.io.load(lpath)

    # Meta data
    _, eeg_id, _ = mngs.gen.split_fpath(lpath)
    meta_all = mngs.io.load(f"./data/Kaggle/{train_or_test}.csv")
    meta = meta_all[meta_all.eeg_id == int(eeg_id)]

    n_subs_uq = len(meta.patient_id.unique())
    if n_subs_uq != 1:
        return None, None

    # Mean Votes across recording
    votes_columns = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]
    if train_or_test == "test":
        for col in votes_columns:
            meta[col] = 1.0
    votes = meta[votes_columns]
    votes = votes.sum() / votes.sum().sum()

    # add eeg_id
    votes["eeg_id"] = eeg_id
    votes["subject_id"] = meta.patient_id.iloc[0]

    # Sort indices
    votes = votes[["subject_id", "eeg_id"] + votes_columns]
    meta = votes

    # Preprocess
    SAMP_RATE_KAGGLE = 200
    raw = to_raw(data, sfreq=SAMP_RATE_KAGGLE)
    preprocessed_raw = preprocess_raw(
        raw
    )  # Here, the signal is downsampled to CONFIG["TGT_FS"]
    eeg_segs = segment_processed_raw(preprocessed_raw)

    meta = pd.concat([meta for _ in range(len(eeg_segs))], axis=1).T
    return eeg_segs, meta


def to_raw(data, sfreq):
    data_np = data.to_numpy().T
    ch_names = list(data.columns)
    ch_types = ["eeg"] * (len(ch_names) - 1) + ["ecg"]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data_np, info, verbose=False)
    return raw


def main():
    import sys

    import matplotlib.pyplot as plt

    # Opening
    CONFIG, sys.stdout, sys.stderr, plt, cc = mngs.gen.start(sys, plt)

    # Training
    LPATHS = glob("./data/Kaggle/train_eegs/*parquet")
    spath_db = mngs.io.increment_version(
        "./data/Kaggle/dbs/", "eeg_data", ".db"
    )
    kaggle_populate_db(LPATHS, spath_db)

    # Test
    LPATHS = glob("./data/Kaggle/test_eegs/*parquet")
    spath_db = mngs.io.increment_version(
        "./data/Kaggle/dbs/", "test_eeg_data", ".db"
    )
    kaggle_populate_db(LPATHS, spath_db)

    # Closing
    mngs.gen.close(CONFIG)


if __name__ == "__main__":
    main()

    # EOF
