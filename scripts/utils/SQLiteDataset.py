#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-01-27 13:35:19 (ywatanabe)"

import re
import sqlite3
from pprint import pprint

import mne
import mngs
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

CONFIG = mngs.io.load("./config/global.yaml")


class SQLiteDataset(Dataset):
    def __init__(self, db_path, show=False):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("SELECT COUNT(*) FROM eeg_data")
        self.length = self.cursor.fetchone()[0]

        # Create a mapping of pseudo IDs to (dataset_id, subject_id) tuples
        self.cursor.execute(
            "SELECT DISTINCT dataset_id, subject_id FROM eeg_data"
        )
        unique_pairs = self.cursor.fetchall()
        self.pseudo_id_map = {
            f"{dataset_id}_{subject_id}": (dataset_id, subject_id)
            for dataset_id, subject_id in unique_pairs
        }

        if show:
            # Load and print unique combinations of dataset_id and subject_id
            self.cursor.execute(
                "SELECT DISTINCT dataset_id, subject_id FROM eeg_data"
            )
            unique_dataset_subject_ids = self.cursor.fetchall()

            # Sort the unique combinations for better readability if needed
            unique_dataset_subject_ids.sort(key=lambda x: (x[0], x[1]))

            # Print the unique combinations
            print("\nLoaded:\n")
            pprint(unique_dataset_subject_ids)
            uq_datasets = np.unique(
                [ds_sub[0] for ds_sub in unique_dataset_subject_ids]
            )
            print(f"\n# of datasets:\n{len(uq_datasets)}")
            print(f"\n# of subjects:\n{len(unique_dataset_subject_ids)}")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Execute the SQL query against the database
        self.cursor.execute(
            "SELECT dataset_id, subject_id, segment FROM eeg_data WHERE id = ?",
            (index + 1,),
        )
        row = self.cursor.fetchone()
        dataset_id, subject_id, segment = row

        # Converts the segment from bytes to a NumPy array and then to a PyTorch tensor
        segment_tensor = torch.from_numpy(
            np.frombuffer(segment, dtype=np.float64)
            .reshape(len(CONFIG["TGT_TRANVERSE_BIPOLAR_MONTAGE"]), -1)
            .copy()
        ).float()
        return dataset_id, subject_id, segment_tensor

    def get_pseudo_ids(self):
        """Retrieve unique pseudo IDs for each subject across datasets."""
        return list(self.pseudo_id_map.keys())

    def get_subset_by_pseudo_ids(self, pseudo_ids):
        """Retrieve a subset of the dataset based on pseudo IDs."""
        indices = []
        for pseudo_id in pseudo_ids:
            dataset_id, subject_id = self.pseudo_id_map[pseudo_id]
            self.cursor.execute(
                "SELECT id FROM eeg_data WHERE dataset_id = ? AND subject_id = ?",
                (dataset_id, subject_id),
            )
            indices.extend(
                [row[0] - 1 for row in self.cursor.fetchall()]
            )  # Adjusting index to be 0-based
        return Subset(self, indices)

    # def get_subject_ids(self):
    #     """Retrieve unique subject IDs from the database."""
    #     self.cursor.execute("SELECT DISTINCT subject_id FROM eeg_data")
    #     subject_ids = [row[0] for row in self.cursor.fetchall()]
    #     return subject_ids

    # def get_subset_by_subjects(self, subject_ids):
    #     """Retrieve a subset of the dataset based on subject IDs."""
    #     placeholders = ",".join("?" for _ in subject_ids)
    #     query = f"SELECT id FROM eeg_data WHERE subject_id IN ({placeholders})"
    #     self.cursor.execute(query, subject_ids)
    #     indices = [
    #         row[0] - 1 for row in self.cursor.fetchall()
    #     ]  # Adjusting index to be 0-based
    #     return Subset(self, indices)

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    import os
    import random
    import sys
    from datetime import datetime

    import matplotlib.pyplot as plt
    import mngs

    CONFIG, sys.stdout, sys.stderr, plt = mngs.gen.start(sys, plt)
    WINDOW_SIZE_PTS = CONFIG["WINDOW_SIZE_SEC"] * CONFIG["TGT_FS"]

    # Loads the database iteratively using pytorch dataloader
    sqlite_dataset = SQLiteDataset(db_path=CONFIG["DB_PATH"], show=True)

    # You might want to pack the dataset to DataLoader
    data_loader = DataLoader(
        sqlite_dataset,
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    ds_all = []
    batch_counter = 0
    for batch in data_loader:
        dataset_ids, subject_ids, X = batch
        # print(X)
        print(X.isnan().any())

        X = X[~X.isnan().any(-1).any(-1)]

        start = random.randint(0, X.shape[-1] - WINDOW_SIZE_PTS)
        end = start + WINDOW_SIZE_PTS
        X = X[..., start:end]  # (64, 14, 256)

        if torch.isnan(X).any():
            raise ValueError

        ds_all.append(dataset_ids)
        batch_counter += 1

    # Don't forget to close the database connection when done
    sqlite_dataset.close()

    # END
    mngs.gen.close(CONFIG)
