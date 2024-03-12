#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-03-09 13:24:58 (ywatanabe)"

from glob import glob

import mngs
from scripts.data.load.preprocess import preprocess_raw, segment_processed_raw
from tqdm import tqdm

CONFIG = mngs.io.load_configs(show=False)
SAMP_RATE = 200

# Functions
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
            metadata TEXT  -- Column to store metadata as JSON string
        )
        """
    )

    count_populated = 0
    subject_ids = []
    for filepath in files_list:
        print(f"\nPopulating: {filepath}")

        # dataset_id, subject_id, is_retrieved = extract_ids(filepath)
        eeg_segments, metadatas = load_kaggle(filepath)
        subject_id = metadatas.subject_id.unique()[0]

        # if not is_retrieved:
        #     logging.warning(
        #         f"dataset_id or subject_id were not extracted from {filepath}"
        #     )
        #     continue

        # # Loads EEG segments
        # eeg_segments, metadatas = load_a_processed_and_segmented_eeg_file(
        #     filepath
        # )

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
            segment_bytes = segment.tobytes()
            cursor.execute(
                "INSERT INTO eeg_data (subject_id, segment_number, segment, metadata) VALUES (?, ?, ?, ?)",
                (subject_id, i_seg, segment_bytes, metadata),
            )

        subject_ids.append(subject_id)
        count_populated += 1

    conn.commit()

    # Print the summary
    cursor.execute("SELECT DISTINCT subject_id FROM eeg_data")
    subject_ids = [row[0] for row in cursor.fetchall()]
    print(subject_ids)
    conn.close()

    n_subjects = len(np.unique(subject_ids))
    perc_populated = round(100 * (count_populated / len(files_list)), 2)
    print(
        f"\nConglaturations! SQLite databse was created from {count_populated} files ({perc_populated}%), "
        f"{n_subjects} subjects:\n{spath_db}"
    )


def load_kaggle(lpath):
    # EEG signals
    data = mngs.io.load(lpath)

    # Meta data
    _, eeg_id, _ = mngs.gen.split_fpath(lpath)
    meta_all = mngs.io.load("./data/Kaggle/train.csv")
    meta = meta_all[meta_all.eeg_id == int(eeg_id)]

    assert len(meta.patient_id.unique()) == 1

    # Mean Votes across recording
    votes_columns = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]
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
    import mne
    import numpy as np
    import pandas as pd

    data_np = data.to_numpy().T
    ch_names = list(data.columns)
    ch_types = ["eeg"] * (len(ch_names) - 1) + ["ecg"]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data_np, info)
    return raw


LPATHS = glob("./data/Kaggle/train_eegs/*parquet")[:10]
# spath_db = "./data/Kaggle/dbs/eeg_data_v001.db"
data_cat = "Kaggle"
spath_db = mngs.io.increment_version(
    CONFIG[f"DIR_{data_cat.upper()}"] + "dbs/", "eeg_data", ".db"
)
kaggle_populate_db(LPATHS, spath_db)


# # Loads

# # lpath = LPATHS[0]
# # data_all = []
# # meta_all = []


# for lpath in tqdm(LPATHS):
#     data, meta = load_kaggle(lpath)
#     import ipdb

#     ipdb.set_trace()
#     data_all.append(data)
#     meta_all.append(meta)

# sample_submission = mngs.io.load("./data/Kaggle/sample_submission.csv")
# print(sample_submission)
# #        eeg_id  seizure_vote  lpd_vote  gpd_vote  lrda_vote  grda_vote  other_vote
# # 0  3911565283      0.166667  0.166667  0.166667   0.166667   0.166667    0.166667
