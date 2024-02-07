#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-06 21:29:48 (ywatanabe)"


import json
import re
import sqlite3
import sys
from collections import OrderedDict
from pprint import pprint

import matplotlib.pyplot as plt
import mne
import mngs
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

CONFIG = mngs.io.load_configs()


class SQLiteDataset(Dataset):
    """
    Note that this SQLiteDataset class works any dataset, regardless of the types of metadata
    """

    def __init__(self, lpath_db, allowed_metadata_mapping={}, show=False):
        self.conn = sqlite3.connect(lpath_db)
        self.cursor = self.conn.cursor()
        self.allowed_metadata_mapping = allowed_metadata_mapping
        self.allowed_indices = self.limit_segments_per_subject(
            self.find_allowed_indices()
        )
        self.length = self.__len__

        if show:
            self.print_database_info()

    def print_database_info(
        self,
    ):
        # Print unique dataset_ids and subject_ids
        self.cursor.execute(
            "SELECT DISTINCT dataset_id, subject_id FROM eeg_data"
        )
        unique_dataset_subject_ids = self.cursor.fetchall()
        unique_dataset_subject_ids.sort(key=lambda x: (x[0], x[1]))
        print("\nLoaded:\n")
        print(pd.DataFrame(np.array(unique_dataset_subject_ids)))
        uq_datasets = np.unique(
            [ds_sub[0] for ds_sub in unique_dataset_subject_ids]
        )
        print(f"\n# of datasets:\n{len(uq_datasets)}")
        print(f"\n# of subjects:\n{len(unique_dataset_subject_ids)}")

    def find_allowed_indices(self):
        # Build the WHERE clause based on the allowed_metadata_mapping
        where_clauses = []
        params = []
        for key, allowed_values in self.allowed_metadata_mapping.items():
            placeholder = ",".join("?" for _ in allowed_values)
            where_clauses.append(
                f"json_extract(metadata, '$.{key}') IN ({placeholder})"
            )
            params.extend(allowed_values)
        where_statement = " AND ".join(where_clauses) if where_clauses else "1"

        # Execute the SQL query against the database with the WHERE clause
        query = f"""
        SELECT id
        FROM eeg_data
        WHERE {where_statement}
        """

        self.cursor.execute(query, params)
        rows = self.cursor.fetchall()
        return [row[0] - 1 for row in rows]

    # fixme: datasplitting should be done outside of this class
    def custom_split(
        self,
        num_folds,
        i_fold,
        data_splitting_criteria,
        undersampling_criteria,
    ):
        """
        Splits the dataset into training, validation, and test datasets based on certain criteria.

        Args:
            num_folds (int): The number of folds for cross-validation.
            i_fold (int): The index of the current fold.
            data_splitting_criteria (list): List of columns to use for data splitting.
            undersampling_criteria (list): List of columns to use for undersampling.

        Returns:
            dict: A dictionary containing 'train', 'val', and 'test' keys with corresponding indices.
        """
        # Extract the criteria values
        criteria_values = [
            tuple(
                self.metadata[idx][criteria]
                for criteria in data_splitting_criteria
            )
            for idx in self.allowed_indices
        ]
        unique_criteria_combinations = list(set(criteria_values))

        # Split the unique combinations into train and test
        train_combinations, test_combinations = train_test_split(
            unique_criteria_combinations, test_size=1.0 / num_folds
        )

        # For cross-validation, further split the train combinations into train and val
        skf = StratifiedKFold(n_splits=num_folds)
        train_val_combinations = list(
            skf.split(train_combinations, [0] * len(train_combinations))
        )[i_fold]
        train_combinations = [
            train_combinations[i] for i in train_val_combinations[0]
        ]
        val_combinations = [
            train_combinations[i] for i in train_val_combinations[1]
        ]

        # Map the combinations back to indices
        train_indices = [
            idx
            for idx, values in enumerate(criteria_values)
            if values in train_combinations
        ]
        val_indices = [
            idx
            for idx, values in enumerate(criteria_values)
            if values in val_combinations
        ]
        test_indices = [
            idx
            for idx, values in enumerate(criteria_values)
            if values in test_combinations
        ]

        # Balance the training data
        train_indices = self.balance_data(
            train_indices, undersampling_criteria
        )

        return {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices,
        }

    # fixme: balance data should be done outside of this class and only applied to the training data
    def balance_data(self, indices, undersampling_criteria):
        """
        Balances the training data based on the specified criteria.

        Args:
            indices (list): The list of indices for the training set.
            undersampling_criteria (list): The list of columns to balance by.

        Returns:
            list: The list of balanced indices.
        """

        # Fetch metadata for the given indices
        metadata = [self.metadata[idx] for idx in indices]

        # Count the number of occurrences for each class based on undersampling criteria
        class_counts = {}
        for data in metadata:
            key = tuple(data[criteria] for criteria in undersampling_criteria)
            class_counts[key] = class_counts.get(key, 0) + 1

        # Find the minimum count to perform undersampling
        min_count = min(class_counts.values())

        # Perform undersampling
        balanced_indices = []
        class_sampled_counts = {key: 0 for key in class_counts}
        for idx, data in zip(indices, metadata):
            key = tuple(data[criteria] for criteria in undersampling_criteria)
            if class_sampled_counts[key] < min_count:
                balanced_indices.append(idx)
                class_sampled_counts[key] += 1

        return balanced_indices

    def __len__(self):
        return len(self.allowed_indices)

    def __getitem__(self, index):
        limited_index = self.allowed_indices[index]

        # Execute the SQL query against the database with the WHERE clause
        where_clauses = []
        params = []
        for key, allowed_values in self.allowed_metadata_mapping.items():
            placeholder = ",".join("?" for _ in allowed_values)
            where_clauses.append(
                f"json_extract(metadata, '$.{key}') IN ({placeholder})"
            )
            params.extend(allowed_values)
        where_statement = " AND ".join(where_clauses)
        query = f"""
        SELECT dataset_id, subject_id, segment, metadata
        FROM eeg_data
        WHERE id = ? AND {where_statement}
        """
        self.cursor.execute(query, [limited_index + 1] + params)
        row = self.cursor.fetchone()

        # If the row does not match the filter, return None or raise an exception
        if row is None:
            return None

        dataset_id, subject_id, segment, metadata_json = row

        # Converts the segment a NumPy array in bytes to a PyTorch tensor in fp32
        segment_tensor = torch.from_numpy(
            np.frombuffer(segment, dtype=np.float64)
            .reshape(len(CONFIG["TGT_TRANVERSE_BIPOLAR_MONTAGE"]), -1)
            .copy()
        ).float()

        # Convert the metadata dictionary to a tensor using dynamic mapping
        metadata_tensor = self.convert_metadata_to_tensor(metadata_json)

        return dataset_id, subject_id, segment_tensor, metadata_tensor

    def convert_metadata_to_tensor(self, metadata_json):
        metadata_dict = json.loads(metadata_json)
        # Assuming metadata is a dictionary with string values
        encoded_features = []
        for key, value in metadata_dict.items():
            encoded_value = self.allowed_metadata_mapping[key][value]
            encoded_features.append(encoded_value)
        return torch.tensor(encoded_features, dtype=torch.long)

    # this should be also outside of this class
    def limit_segments_per_subject(self, indices):
        # Group indices by subject
        subject_to_indices = {}
        for index in indices:
            self.cursor.execute(
                "SELECT subject_id FROM eeg_data WHERE id = ?", (index + 1,)
            )
            subject_id = self.cursor.fetchone()[0]
            if subject_id not in subject_to_indices:
                subject_to_indices[subject_id] = []
            subject_to_indices[subject_id].append(index)

        # Randomly select up to MAX_N_WINDOWS_PER_SUB segments per subject
        max_segments_per_subject = CONFIG["MAX_N_WINDOWS_PER_SUB"]
        limited_indices = []
        for subject_id, subject_indices in subject_to_indices.items():
            if len(subject_indices) > max_segments_per_subject:
                limited_indices.extend(
                    random.sample(subject_indices, max_segments_per_subject)
                )
            else:
                limited_indices.extend(subject_indices)

        return limited_indices

    # this should be also outside of this class
    def create_dataloaders(
        self, batch_size, num_workers=4, shuffle_train=True
    ):
        """
        Create DataLoaders for training, validation, and testing.

        Parameters:
        - batch_size: The batch size for the DataLoaders.
        - num_workers: The number of subprocesses to use for data loading.
        - shuffle_train: Whether to shuffle the training data.

        Returns:
        - dl_tra: DataLoader for the training set.
        - dl_val: DataLoader for the validation set.
        - dl_tes: DataLoader for the test set.
        """
        # Perform custom data splitting
        split_indices = self.custom_split(
            num_folds=5,
            i_fold=0,
            data_splitting_criteria=["disease_type", "cognitive_level"],
            undersampling_criteria=["disease_type"],
        )

        # Create DataLoaders for each split
        dl_tra = DataLoader(
            Subset(self, split_indices["train"]),
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            drop_last=True,
        )
        dl_val = DataLoader(
            Subset(self, split_indices["val"]),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )
        dl_tes = DataLoader(
            Subset(self, split_indices["test"]),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )

        return dl_tra, dl_val, dl_tes

    # def get_dataset_ids(self):
    #     """Retrieve unique dataset IDs from the database."""
    #     self.cursor.execute("SELECT DISTINCT dataset_id FROM eeg_data")
    #     dataset_ids = [row[0] for row in self.cursor.fetchall()]
    #     return dataset_ids

    # def get_subset_by_dataset_ids(self, dataset_ids):
    #     """Retrieve a subset of the dataset based on dataset IDs."""
    #     placeholders = ",".join("?" for _ in dataset_ids)
    #     query = f"SELECT id FROM eeg_data WHERE dataset_id IN ({placeholders})"
    #     self.cursor.execute(query, dataset_ids)
    #     indices = [
    #         row[0] - 1 for row in self.cursor.fetchall()
    #     ]  # Adjusting index to be 0-based
    #     return Subset(self, indices)

    def close(self):
        self.conn.close()


def gen_metadata_mapping(lpath_db):
    """
    Generates independent mappings for each metadata tag from a SQLite database.

    Parameters:
    - lpath_db (str): The file path to the SQLite database.

    Returns:
    - metadata_mapping (OrderedDict of OrderedDict): An ordered dictionary containing mappings for each metadata tag.
    """
    print(f"\nGenerating metadata mapping...\n")

    # Fetch unique metadata values
    conn = sqlite3.connect(lpath_db)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT metadata FROM eeg_data")
    unique_metadata_values = cursor.fetchall()
    conn.close()

    # Initialize an OrderedDict to collect values for each metadata tag
    metadata_values = OrderedDict()

    # Iterate over unique metadata values and populate the metadata_values dictionary
    for (json_string,) in unique_metadata_values:
        metadata = json.loads(json_string)
        for key in metadata:
            if key not in metadata_values:
                metadata_values[key] = set()
            metadata_values[key].add(metadata[key])

    # Create independent mappings for each metadata tag using OrderedDict
    metadata_mapping = OrderedDict()
    for key, values in metadata_values.items():
        metadata_mapping[key] = OrderedDict(
            (value, index) for index, value in enumerate(sorted(values))
        )

    # Print the metadata mapping
    print("\nMetadata mapping:")
    for metadata_key, mapping in metadata_mapping.items():
        print(f"\n{metadata_key.capitalize()}:")
        for value, index in mapping.items():
            print(f"  {value} (Index: {index})")

    return metadata_mapping


def allow_data(metadata_mapping, allow_dict):
    """
    Allows the metadata mapping based on the allowed values and maximum number of subjects.

    Parameters:
    - metadata_mapping (dict): The original metadata mapping.
    - allow_dict (dict): A dictionary specifying allowed values for each metadata field.

    Returns:
    - allowed_metadata_mapping (dict): The limited metadata mapping.
    """
    allowed_metadata_mapping = {}
    for key, mapping in metadata_mapping.items():
        if key in allow_dict:
            # Filter the mapping based on the allowed values
            allowed_values = allow_dict[key]
            limited_mapping = {
                value: idx
                for value, idx in mapping.items()
                if value in allowed_values
            }
            allowed_metadata_mapping[key] = limited_mapping
        else:
            # If the key is not in the allow_dict, copy the mapping as is
            allowed_metadata_mapping[key] = mapping

    return allowed_metadata_mapping


def apply_sliding_window_data_augmentation(segment_tensor):
    WINDOW_SIZE_PTS = int(
        CONFIG["WINDOW_SIZE_SEC"] * CONFIG["TGT_FS"] / CONFIG["OVERLAP_FACTOR"]
    )
    start = random.randint(0, segment_tensor.shape[-1] - WINDOW_SIZE_PTS)
    end = start + WINDOW_SIZE_PTS
    segment_tensor = segment_tensor[..., start:end]
    return segment_tensor


if __name__ == "__main__":
    import os
    import random
    import sys
    from datetime import datetime

    import matplotlib.pyplot as plt
    import mngs

    CONFIG, sys.stdout, sys.stderr, plt, cc = mngs.gen.start(sys, plt)
    WINDOW_SIZE_PTS = CONFIG["WINDOW_SIZE_SEC"] * CONFIG["TGT_FS"]
    mngs.gen.fix_seeds(np=np, random=random, os=os)

    # Loads the database iteratively using pytorch dataloader
    lpath_db = mngs.gen.find_latest(
        "./data/Dementia/dbs/checked", "eeg_data", ".db"
    )

    # Generate all data mapping
    metadata_mapping_all = gen_metadata_mapping(lpath_db)
    # Please hardcode your filter for data used in the following processes
    allow_dict = {
        "disease_type": ["AD", "DLB", "NPH", "HV"],
        "cognitive_level": ["HV", "Dementia", "MCI", "MCI23"],
        "institution": ["Osaka"],
    }
    metadata_mapping_limited = allow_data(metadata_mapping_all, allow_dict)

    # Define the criteria for data splitting and balancing
    data_splitting_criteria = ["dataset_id", "subject_id"]
    undersampling_criteria = ["disease_type"]

    ds = SQLiteDataset(
        lpath_db=lpath_db,
        allowed_metadata_mapping=metadata_mapping_limited,
        show=False,
    )

    # You might want to pack the dataset to DataLoader
    dl = DataLoader(
        ds,
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=True,
        num_workers=4,
        drop_last=True,
        # collate_fn=collate_fn,
    )

    # Main
    D_epoch = []
    S_epoch = []
    M_epoch = []
    for i_batch, batch in enumerate(dl):
        """
        batch = next(iter(dl))
        """
        # Batched dataset ids, subject ids, EEG signals, and metadatas
        Db, Sb, Xb, Mb = batch
        Xb = apply_sliding_window_data_augmentation(Xb)
        D_epoch.append(Db)
        S_epoch.append(Sb)
        M_epoch.append(Mb)

    # Don't forget to close the database connection when done
    ds.close()

    mngs.gen.unique(np.vstack(D_epoch))
    mngs.gen.unique(np.vstack(S_epoch))
    mngs.gen.unique(np.vstack(M_epoch)[:, 0])
    mngs.gen.unique(np.vstack(M_epoch)[:, 1])
    mngs.gen.unique(np.vstack(M_epoch)[:, 2])

    # END
    mngs.gen.close(CONFIG)
