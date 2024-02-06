#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-06 17:59:11 (ywatanabe)"

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
from torch.utils.data import DataLoader, Dataset, Subset

CONFIG = mngs.io.load_configs()


def limit_data(metadata_mapping, allow_dict, max_n_subjects):
    """
    Limits the metadata mapping based on the allowed values and maximum number of subjects.

    Parameters:
    - metadata_mapping (dict): The original metadata mapping.
    - allow_dict (dict): A dictionary specifying allowed values for each metadata field.
    - max_n_subjects (int): Maximum number of subjects to include.

    Returns:
    - limited_metadata_mapping (dict): The limited metadata mapping.
    """
    limited_metadata_mapping = {}
    for key, mapping in metadata_mapping.items():
        if key in allow_dict:
            # Filter the mapping based on the allowed values
            allowed_values = allow_dict[key]
            limited_mapping = {
                value: idx
                for value, idx in mapping.items()
                if value in allowed_values
            }
            limited_metadata_mapping[key] = limited_mapping
        else:
            # If the key is not in the allow_dict, copy the mapping as is
            limited_metadata_mapping[key] = mapping

    # # If there's a subject_id key, limit the number of subjects
    # if "subject_id" in limited_metadata_mapping and max_n_subjects is not None:
    #     subject_ids = list(limited_metadata_mapping["subject_id"].keys())
    #     if len(subject_ids) > max_n_subjects:
    #         # Randomly select a subset of subject IDs if there are too many
    #         selected_subject_ids = set(
    #             random.sample(subject_ids, max_n_subjects)
    #         )
    #         limited_metadata_mapping["subject_id"] = {
    #             sid: idx
    #             for sid, idx in limited_metadata_mapping["subject_id"].items()
    #             if sid in selected_subject_ids
    #         }

    return limited_metadata_mapping


def gen_metadata_mapping(lpath_db):
    """
    Generates independent mappings for each metadata tag from a SQLite database.

    Parameters:
    - lpath_db (str): The file path to the SQLite database.

    Returns:
    - metadata_mapping (OrderedDict of OrderedDict): An ordered dictionary containing mappings for each metadata tag.
    """
    print(f"\nGenerating metadata mapping...\n")
    # Connect to the SQLite database
    conn = sqlite3.connect(lpath_db)
    cursor = conn.cursor()

    # Execute the query to select distinct metadata
    cursor.execute("SELECT DISTINCT metadata FROM eeg_data")

    # Fetch all unique metadata values
    unique_metadata_values = cursor.fetchall()

    # Close the connection to the database
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

    # Nested function to print the metadata mapping
    def print_metadata_mapping(metadata_mapping):
        print("\nMetadata mapping:")
        for metadata_key, mapping in metadata_mapping.items():
            print(f"\n{metadata_key.capitalize()}:")
            for value, index in mapping.items():
                print(f"  {value} (Index: {index})")

    # Call the nested function to print the metadata mapping
    print_metadata_mapping(metadata_mapping)

    return metadata_mapping


class SQLiteDataset(Dataset):
    def __init__(self, lpath_db, limited_metadata_mapping={}, show=False):
        self.conn = sqlite3.connect(lpath_db)
        self.cursor = self.conn.cursor()
        # self.cursor.execute("SELECT COUNT(*) FROM eeg_data") # is this needed?
        # self.length = self.cursor.fetchone()[0] # is this needed?

        self.limited_metadata_mapping = limited_metadata_mapping
        # Calculate the valid indices based on the filter
        self.valid_indices = self.calculate_valid_indices()
        self.length = self.__len__

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

            print(pd.DataFrame(np.array(unique_dataset_subject_ids)))
            uq_datasets = np.unique(
                [ds_sub[0] for ds_sub in unique_dataset_subject_ids]
            )
            print(f"\n# of datasets:\n{len(uq_datasets)}")
            print(f"\n# of subjects:\n{len(unique_dataset_subject_ids)}")

    # how can I debug this function?
    def calculate_valid_indices(self):
        # Build the WHERE clause based on the limited_metadata_mapping
        where_clauses = []
        params = []
        for key, allowed_values in self.limited_metadata_mapping.items():
            placeholder = ",".join("?" for _ in allowed_values)
            where_clauses.append(
                f"json_extract(metadata, '$.{key}') IN ({placeholder})"
            )
            params.extend(allowed_values)

        # Combine the WHERE clauses with AND
        where_statement = " AND ".join(where_clauses) if where_clauses else "1"

        # Execute the SQL query against the database with the WHERE clause
        query = f"""
        SELECT id
        FROM eeg_data
        WHERE {where_statement}
        """

        """
        self = ds
        
        # In [17]: query
        # Out[17]: "\nSELECT id\nFROM eeg_data\nWHERE json_extract(metadata, '$.disease_type') IN (?,?,?,?) AND json_extract(metadata, '$.cognitive_level') IN (?,?,?,?) AND json_extract(metadata, '$.institution') IN (?)\n"

        # In [18]: params
        # Out[18]: 
        # ['AD',
        #  'DLB',
        #  'HV',
        #  'NPH',
        #  'Dementia',
        #  'HV',
        #  'MCI',
        #  'MCI23',
        #  'Osaka University']

        self.cursor.execute(query, params)
        rows = self.cursor.fetchall()

        rows[0] # (1,)
        """

        # In [15]: query
        # Out[15]: "\nSELECT id\nFROM eeg_data\nWHERE json_extract(metadata, '$.disease_type') IN (?,?,?,?) AND json_extract(metadata, '$.cognitive_level') IN (?,?,?,?) AND json_extract(metadata, '$.institution') IN (?)\n"
        # params
        # ['AD',
        #  'DLB',
        #  'HV',
        #  'NPH',
        #  'Dementia',
        #  'HV',
        #  'MCI',
        #  'MCI23',
        #  'Osaka University']

        self.cursor.execute(query, params)
        rows = self.cursor.fetchall()

        # Return the list of valid indices (adjusted to be 0-based)
        return [row[0] - 1 for row in rows]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):
        # Map the requested index to the valid index
        valid_index = self.valid_indices[index]

        try:
            # Build the WHERE clause based on the limited_limited_metadata_mapping
            where_clauses = []
            params = []
            for key, allowed_values in self.limited_metadata_mapping.items():
                placeholder = ",".join("?" for _ in allowed_values)
                where_clauses.append(
                    f"json_extract(metadata, '$.{key}') IN ({placeholder})"
                )
                params.extend(allowed_values)

            # Combine the WHERE clauses with AND
            where_statement = " AND ".join(where_clauses)

            # Execute the SQL query against the database with the WHERE clause
            query = f"""
            SELECT dataset_id, subject_id, segment, metadata
            FROM eeg_data
            WHERE id = ? AND {where_statement}
            """

            """
            self = ds

            # In [10]:             query
            # Out[10]: "\nSELECT dataset_id, subject_id, segment, metadata\nFROM eeg_data\nWHERE id = ? AND json_extract(metadata, '$.disease_type') IN (?,?,?,?) AND json_extract(metadata, '$.cognitive_level') IN (?,?,?,?) AND json_extract(metadata, '$.institution') IN (?)\n"

            # In [11]:             params
            # Out[11]: 
            # ['AD',
            # 'DLB',
            # 'HV',
            # 'NPH',
            # 'Dementia',
            # 'HV',
            # 'MCI',
            # 'MCI23',
            # 'Osaka University']

            ds.valid_indices
            """

            self.cursor.execute(query, [valid_index + 1] + params)
            row = self.cursor.fetchone()

            # If the row does not match the filter, return None or raise an exception
            if row is None:
                return None  # or raise an exception

            dataset_id, subject_id, segment, metadata_json = row

            # Converts the segment from bytes to a NumPy array and then to a PyTorch tensor
            segment_tensor = torch.from_numpy(
                np.frombuffer(segment, dtype=np.float64)
                .reshape(len(CONFIG["TGT_TRANVERSE_BIPOLAR_MONTAGE"]), -1)
                .copy()
            ).float()

            # Convert the metadata dictionary to a tensor using dynamic mapping
            metadata_dict = json.loads(metadata_json)
            metadata_tensor = self.convert_metadata_to_tensor(metadata_dict)

            return dataset_id, subject_id, segment_tensor, metadata_tensor
        except Exception as e:
            print(f"Error fetching data for index {index}: {e}")
            # Optionally, you can return a placeholder value or raise the exception
            raise

    def convert_metadata_to_tensor(self, metadata):
        # Assuming metadata is a dictionary with string values
        encoded_features = []
        for key, value in metadata.items():
            encoded_value = self.limited_metadata_mapping[key][value]
            encoded_features.append(encoded_value)
        return torch.tensor(encoded_features, dtype=torch.long)

    def get_dataset_ids(self):
        """Retrieve unique dataset IDs from the database."""
        self.cursor.execute("SELECT DISTINCT dataset_id FROM eeg_data")
        dataset_ids = [row[0] for row in self.cursor.fetchall()]
        return dataset_ids

    def get_subset_by_dataset_ids(self, dataset_ids):
        """Retrieve a subset of the dataset based on dataset IDs."""
        placeholders = ",".join("?" for _ in dataset_ids)
        query = f"SELECT id FROM eeg_data WHERE dataset_id IN ({placeholders})"
        self.cursor.execute(query, dataset_ids)
        indices = [
            row[0] - 1 for row in self.cursor.fetchall()
        ]  # Adjusting index to be 0-based
        return Subset(self, indices)

    def close(self):
        self.conn.close()


def test_ds(ds):
    # Then, try to iterate over the dataset manually to see if any index raises an error
    for i in range(len(ds)):
        item = ds[i]
        assert item is not None


if __name__ == "__main__":
    import os
    import random
    import sys
    from datetime import datetime

    import matplotlib.pyplot as plt
    import mngs

    CONFIG, sys.stdout, sys.stderr, plt, cc = mngs.gen.start(sys, plt)
    WINDOW_SIZE_PTS = CONFIG["WINDOW_SIZE_SEC"] * CONFIG["TGT_FS"]

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
        # "institution": ['ds_BIDS_dataset_Nissei_v1']
    }
    max_n_subjects = 100  # For example, limit to 100 subjects
    metadata_mapping_limited = limit_data(
        metadata_mapping_all, allow_dict, max_n_subjects
    )

    ds = SQLiteDataset(
        lpath_db=lpath_db,
        limited_metadata_mapping=metadata_mapping_limited,
        show=False,
    )

    # You might want to pack the dataset to DataLoader
    dl = DataLoader(
        ds,
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    # print(ds.limited_metadata_mapping)
    print(dl.dataset.limited_metadata_mapping)

    # koko
    ds_all = []
    for i_batch, batch in enumerate(dl):
        """
        batch = next(iter(dl))
        """
        dataset_ids, subject_ids, X, meta = batch

        X = X[~X.isnan().any(-1).any(-1)]

        start = random.randint(0, X.shape[-1] - WINDOW_SIZE_PTS)
        end = start + WINDOW_SIZE_PTS
        X = X[..., start:end]  # (64, 14, 256)

        if torch.isnan(X).any():
            raise ValueError

        ds_all.append(dataset_ids)

    # Don't forget to close the database connection when done
    ds.close()

    print(np.unique(ds_all))
    # END
    mngs.gen.close(CONFIG)
