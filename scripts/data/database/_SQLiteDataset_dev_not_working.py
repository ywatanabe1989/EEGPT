#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-07 08:07:50 (ywatanabe)"


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


class SQLiteDataset(Dataset):
    """
    Note that this SQLiteDataset class works any dataset, regardless of the types of metadata
    """

    def __init__(self, lpath_db, show=False):
        self.lpath_db = lpath_db
        self.conn = sqlite3.connect(lpath_db)
        self.cursor = self.conn.cursor()
        self.inclusion_criteria = None

        if show:
            self.print_database()

    def apply_inclusion_criteria(self, inclusion_criteria):
        self.inclusion_criteria = inclusion_criteria
        # Prepare SQL query parts for filtering
        where_clauses = []
        params = []

        # Filter by dataset_id and subject_id
        if inclusion_criteria.get("dataset_id"):
            placeholder = ",".join(
                "?" for _ in inclusion_criteria["dataset_id"]
            )
            where_clauses.append(f"dataset_id IN ({placeholder})")
            params.extend(inclusion_criteria["dataset_id"])

        if inclusion_criteria.get("subject_id"):
            placeholder = ",".join(
                "?" for _ in inclusion_criteria["subject_id"]
            )
            where_clauses.append(f"subject_id IN ({placeholder})")
            params.extend(inclusion_criteria["subject_id"])

        # Filter by metadata
        metadata_criteria = inclusion_criteria.get("metadata", {})
        for key, values in metadata_criteria.items():
            # Ensure that the JSON path is correctly specified
            json_extract_query = " OR ".join(
                f"json_extract(metadata, '$.{key}') = ?" for _ in values
            )
            where_clauses.append(f"({json_extract_query})")
            params.extend(values)

        # Combine all where clauses
        where_statement = " AND ".join(where_clauses) if where_clauses else "1"

        # Execute the SQL query against the database with the WHERE clause
        query = f"""
        SELECT id
        FROM eeg_data
        WHERE {where_statement}
        """
        self.cursor.execute(query, params)
        rows = self.cursor.fetchall()

        # Extract the ids of the rows that match the filter
        self.allowed_indices = [
            row[0] - 1 for row in rows
        ]  # Assuming 'id' is 1-indexed in the database

        # Update the metadata property based on the filtered indices
        self._update_metadata()

    # def apply_inclusion_criteria(self, inclusion_criteria):
    #     self.inclusion_criteria = inclusion_criteria
    #     # Prepare SQL query parts for filtering
    #     where_clauses = []
    #     params = []

    #     # Filter by dataset_id and subject_id
    #     if inclusion_criteria.get("dataset_id"):
    #         placeholder = ",".join(
    #             "?" for _ in inclusion_criteria["dataset_id"]
    #         )
    #         where_clauses.append(f"dataset_id IN ({placeholder})")
    #         params.extend(inclusion_criteria["dataset_id"])

    #     if inclusion_criteria.get("subject_id"):
    #         placeholder = ",".join(
    #             "?" for _ in inclusion_criteria["subject_id"]
    #         )
    #         where_clauses.append(f"subject_id IN ({placeholder})")
    #         params.extend(inclusion_criteria["subject_id"])

    #     # Filter by metadata
    #     metadata_criteria = inclusion_criteria.get("metadata", {})
    #     for key, values in metadata_criteria.items():
    #         json_extract_query = " OR ".join(
    #             f"json_extract(metadata, '$.{key}') = ?" for _ in values
    #         )
    #         where_clauses.append(f"({json_extract_query})")
    #         params.extend(values)

    #     # Combine all where clauses
    #     where_statement = " AND ".join(where_clauses) if where_clauses else "1"

    #     # Execute the SQL query against the database with the WHERE clause
    #     query = f"""
    #     SELECT id
    #     FROM eeg_data
    #     WHERE {where_statement}
    #     """
    #     self.cursor.execute(query, params)
    #     rows = self.cursor.fetchall()

    #     # Extract the ids of the rows that match the filter
    #     self.allowed_indices = [
    #         row[0] - 1 for row in rows
    #     ]  # Assuming 'id' is 1-indexed in the database

    #     # Update the metadata property based on the filtered indices
    #     self._update_metadata()

    def _update_metadata(self):
        # This method should update the metadata property based on the filtered indices (self.allowed_indices)
        allowed_metadata = []
        for index in self.allowed_indices:
            self.cursor.execute(
                "SELECT metadata FROM eeg_data WHERE id = ?", (index + 1,)
            )
            row = self.cursor.fetchone()
            if row:
                metadata_json = row[0]
                metadata_dict = json.loads(metadata_json)
                allowed_metadata.append(metadata_dict)

        # Now we have a list of metadata dictionaries for the allowed indices
        # We need to create a mapping for each metadata key to the allowed values
        allowed_metadata_mapping = OrderedDict()
        for metadata_dict in allowed_metadata:
            for key, value in metadata_dict.items():
                if key not in allowed_metadata_mapping:
                    allowed_metadata_mapping[key] = set()
                allowed_metadata_mapping[key].add(value)

        # Convert sets to sorted lists for consistent ordering
        for key in allowed_metadata_mapping:
            allowed_metadata_mapping[key] = sorted(
                list(allowed_metadata_mapping[key])
            )

        # Update the metadata property
        self._metadata = allowed_metadata_mapping

    @property
    def metadata(self):
        """
        Retrieves the metadata mapping after applying inclusion criteria or generates initial mapping.

        Returns:
        - dict: The metadata mapping.
        """
        if hasattr(self, "_metadata"):
            return self._metadata  # [REVISED]
        else:
            return self._generate_initial_metadata_mapping()  # [REVISED]

    def _generate_initial_metadata_mapping(self):
        """
        Generates the initial metadata mapping from the SQLite database.

        Returns:
        - OrderedDict: The initial metadata mapping.
        """
        conn = sqlite3.connect(self.lpath_db)
        cursor = conn.cursor()
        metadata_val_uq = cursor.execute(
            "SELECT DISTINCT metadata FROM eeg_data"
        ).fetchall()
        dataset_id_uq = [
            item[0]
            for item in cursor.execute(
                "SELECT DISTINCT dataset_id FROM eeg_data"
            ).fetchall()
        ]
        subject_id_uq = [
            item[0]
            for item in cursor.execute(
                "SELECT DISTINCT subject_id FROM eeg_data"
            ).fetchall()
        ]
        conn.close()

        metadata_values = OrderedDict(
            [
                ("dataset_id", dataset_id_uq),
                ("subject_id", subject_id_uq),
            ]
        )

        for (json_string,) in metadata_val_uq:
            metadata = json.loads(json_string)
            for key in metadata:
                if key not in metadata_values:
                    metadata_values[key] = set()
                metadata_values[key].add(metadata[key])

        metadata_mapping = OrderedDict()
        for key, values in metadata_values.items():
            metadata_mapping[key] = OrderedDict(
                (value, index) for index, value in enumerate(sorted(values))
            )

        return metadata_mapping  # [REVISED]

    def _generate_allowed_metadata_mapping(self):
        # This method should generate the allowed_metadata_mapping based on the filtered indices (self.allowed_indices)
        # Assuming that the metadata is stored as a JSON string in the 'metadata' column
        allowed_metadata = []
        for index in self.allowed_indices:
            self.cursor.execute(
                "SELECT metadata FROM eeg_data WHERE id = ?", (index + 1,)
            )
            row = self.cursor.fetchone()
            if row:
                metadata_json = row[0]
                metadata_dict = json.loads(metadata_json)
                allowed_metadata.append(metadata_dict)

        # Now we have a list of metadata dictionaries for the allowed indices
        # We need to create a mapping for each metadata key to the allowed values
        allowed_metadata_mapping = OrderedDict()
        for metadata_dict in allowed_metadata:
            for key, value in metadata_dict.items():
                if key not in allowed_metadata_mapping:
                    allowed_metadata_mapping[key] = set()
                allowed_metadata_mapping[key].add(value)

        # Convert sets to sorted lists for consistent ordering
        for key in allowed_metadata_mapping:
            allowed_metadata_mapping[key] = sorted(
                list(allowed_metadata_mapping[key])
            )

        return allowed_metadata_mapping

    def print_database(self):
        pass
        # self.cursor.execute(
        #     "SELECT DISTINCT dataset_id, subject_id FROM eeg_data"
        # )
        # unique_dataset_subject_ids = self.cursor.fetchall()
        # unique_dataset_subject_ids.sort(key=lambda x: (x[0], x[1]))
        # print("\nLoaded:\n")
        # print(pd.DataFrame(np.array(unique_dataset_subject_ids)))
        # uq_datasets = np.unique(
        #     [ds_sub[0] for ds_sub in unique_dataset_subject_ids]
        # )
        # print(f"\n# of datasets:\n{len(uq_datasets)}")
        # print(f"\n# of subjects:\n{len(unique_dataset_subject_ids)}")

    def __getitem__(self, index):
        limited_index = self.allowed_indices[index]

        # Execute the SQL query against the database with the WHERE clause
        where_clauses = []
        params = []
        for (
            key,
            allowed_values,
        ) in (
            self.inclusion_criteria.items()
        ):  # self.allowed_metadata_mapping is not created yet
            placeholder = ",".join("?" for _ in allowed_values)
            where_clauses.append(
                f"json_extract(metadata, '$.{key}') IN ({placeholder})"
            )
            params.extend(allowed_values)

        # Add the inclusion criteria to the WHERE clause if it exists
        if self.inclusion_criteria:
            for (
                criterion_key,
                criterion_value,
            ) in self.inclusion_criteria.items():
                where_clauses.append(f"{criterion_key} = ?")
                params.append(criterion_value)

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
        metadata_dict = dict(dataset_id=dataset_id, subject_id=subject_id)
        metadata_dict.update(json.loads(metadata_json))

        return dataset_id, subject_id, segment_tensor, metadata_dict

    def close(self):
        self.conn.close()


# def apply_inclusion_criteria(self, inclusion_criteria):
#     """Applies inclusion criteria and generates a new subset which meets the requirements.

#     Args:
#         inclusion_criteria (dict): The criteria to apply, with keys for each field to filter on.

#     Returns:
#         SQLiteDataset: A new SQLiteDataset instance representing the filtered dataset.
#     """
#     # Prepare SQL query parts for filtering
#     where_clauses = []
#     params = []

#     # Filter by dataset_id and subject_id
#     if inclusion_criteria.get("dataset_id"):
#         placeholder = ",".join("?" for _ in inclusion_criteria["dataset_id"])
#         where_clauses.append(f"dataset_id IN ({placeholder})")
#         params.extend(inclusion_criteria["dataset_id"])

#     if inclusion_criteria.get("subject_id"):
#         placeholder = ",".join("?" for _ in inclusion_criteria["subject_id"])
#         where_clauses.append(f"subject_id IN ({placeholder})")
#         params.extend(inclusion_criteria["subject_id"])

#     # Filter by metadata
#     metadata_criteria = inclusion_criteria.get("metadata", {})
#     for key, values in metadata_criteria.items():
#         json_extract_query = " OR ".join(
#             f"json_extract(metadata, '$.{key}') = ?" for _ in values
#         )
#         where_clauses.append(f"({json_extract_query})")
#         params.extend(values)

#     # Combine all where clauses
#     where_statement = " AND ".join(where_clauses) if where_clauses else "1"

#     # Execute the SQL query against the database with the WHERE clause
#     query = f"""
#     SELECT id
#     FROM eeg_data
#     WHERE {where_statement}
#     """
#     self.cursor.execute(query, params)
#     rows = self.cursor.fetchall()

#     # Extract the ids of the rows that match the filter
#     filtered_ids = [row[0] for row in rows]

#     # Create a new SQLiteDataset instance with the filtered ids
#     filtered_dataset = SQLiteDataset(self.lpath_db)

#     # filtered_dataset.allowed_indices = filtered_ids
#     # filtered_dataset.allowed_metadata_mapping = self.metadata

#     return filtered_dataset


# def apply_inclusion_criteria(ds, inclusion_criteria):
#     """Applies inclusino criteria and generate new subset which meets the requirements."""
#     # pass
#     return filtered_ds


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
    ds = SQLiteDataset(
        lpath_db=lpath_db,
        show=False,
    )
    ds.metadata

    inclusion_criteria = {
        "dataset_id": [],  # any
        "subject_id": [],
        "metadata": {
            "disease_type": ["AD", "DLB", "NPH", "HV"],
            "cognitive_level": ["HV", "Dementia", "MCI", "MCI23"],
            "institution": ["Osaka"],
        },
    }

    # why HV is not included??
    ds.apply_inclusion_criteria(inclusion_criteria)
    ds.metadata
    ds[0]

    # In [46]:     ds.metadata  # this include institutions like Nissei and Kochi, which are outside of inclusion_cri
    #     ...: teria
    # Out[46]:
    # OrderedDict([('disease_type', ['AD', 'DLB', 'NPH']),
    #              ('cognitive_level', ['Dementia', 'MCI', 'MCI23']),
    #              ('institution', ['Osaka'])])

    # koko

    # # # Generate all data mapping
    # # metadata_mapping_all = map_metadata(lpath_db)

    # # Please hardcode your filter for data used in the following processes
    # inclusion_criteria = {
    #     "dataset_id": ["ds-0001", "ds-0002"],  # any
    #     "subject_id": ["sub-0001", "sub-0002"],
    #     "disease_type": ["AD", "DLB", "NPH", "HV"],
    #     "cognitive_level": ["HV", "Dementia", "MCI", "MCI23"],
    #     "institution": ["Osaka"],
    # }
    # # metadata_mapping, _ = apply_inclusion_criteria(
    # #     metadata_mapping_all, inclusion_criteria
    # # )
    # # print(metadat_mapping)
    # # {'disease_type': {'AD': 0, 'DLB': 3, 'HV': 7, 'NPH': 9},
    # #  'cognitive_level': {'Dementia': 0, 'HV': 1, 'MCI': 2, 'MCI23': 3},
    # #  'institution': {'Osaka': 2}}

    # # how can I consider dataset_id and subject_id here?? metadata_mapping should include all inclusion criteria

    # ds = SQLiteDataset(
    #     lpath_db=lpath_db,
    #     allowed_metadata_mapping=metadata_mapping_limited,
    #     show=False,
    # )

    # # You might want to pack the dataset to DataLoader
    # dl = DataLoader(
    #     ds,
    #     batch_size=CONFIG["BATCH_SIZE"],
    #     shuffle=True,
    #     num_workers=4,
    #     drop_last=True,
    #     # collate_fn=collate_fn,
    # )

    # # Main
    # D_epoch = []
    # S_epoch = []
    # M_epoch = []
    # for i_batch, batch in enumerate(dl):
    #     """
    #     batch = next(iter(dl))
    #     """
    #     # Batched dataset ids, subject ids, EEG signals, and metadatas
    #     Db, Sb, Xb, Mb = batch
    #     Xb = apply_sliding_window_data_augmentation(Xb)
    #     D_epoch.append(Db)
    #     S_epoch.append(Sb)
    #     M_epoch.append(Mb)

    # # Don't forget to close the database connection when done
    # ds.close()

    # mngs.gen.unique(np.vstack(D_epoch))
    # mngs.gen.unique(np.vstack(S_epoch))
    # mngs.gen.unique(np.vstack(M_epoch)[:, 0])
    # mngs.gen.unique(np.vstack(M_epoch)[:, 1])
    # mngs.gen.unique(np.vstack(M_epoch)[:, 2])

    # # END
    # mngs.gen.close(CONFIG)

    # def find_inclusion_indice(self):
    #     # Build the WHERE clause based on the allowed_metadata_mapping
    #     where_clauses = []
    #     params = []
    #     for key, allowed_values in self.allowed_metadata_mapping.items():
    #         if key in ["dataset_id", "subject_id"]:
    #             # Handle dataset_id and subject_id as special cases
    #             if (
    #                 allowed_values is not None
    #             ):  # Check if specific values are provided
    #                 placeholder = ",".join("?" for _ in allowed_values)
    #                 where_clauses.append(f"{key} IN ({placeholder})")
    #                 params.extend(allowed_values)
    #         else:
    #             # Handle other metadata fields
    #             placeholder = ",".join("?" for _ in allowed_values)
    #             where_clauses.append(
    #                 f"json_extract(metadata, '$.{key}') IN ({placeholder})"
    #             )
    #             params.extend(allowed_values)
    #     where_statement = " AND ".join(where_clauses) if where_clauses else "1"

    #     # Execute the SQL query against the database with the WHERE clause
    #     query = f"""
    #     SELECT id
    #     FROM eeg_data
    #     WHERE {where_statement}
    #     """

    #     self.cursor.execute(query, params)
    #     rows = self.cursor.fetchall()
    #     return [row[0] - 1 for row in rows]

    # here, dataset_id and subject_id are string. Is it possible to keep metadata as  string??

    # def convert_metadata_to_tensor(self, metadata_json):
    #     metadata_dict = json.loads(metadata_json)
    #     # Assuming metadata is a dictionary with string values
    #     encoded_features = []
    #     for key, value in metadata_dict.items():
    #         encoded_value = self.allowed_metadata_mapping[key][value]
    #         encoded_features.append(encoded_value)
    #     return torch.tensor(encoded_features, dtype=torch.long)


# def apply_inclusion_criteria(metadata_mapping, inclusion_criteria):
#     """
#     Allows the metadata mapping based on the allowed values and maximum number of subjects.

#     Parameters:
#     - metadata_mapping (dict): The original metadata mapping.
#     - inclusion_criteria (dict): A dictionary specifying allowed values for each metadata field.

#     Returns:
#     - allowed_metadata_mapping (dict): The limited metadata mapping.
#     - dataset_subject_criteria (dict): The inclusion criteria for dataset_id and subject_id.
#     """
#     allowed_metadata_mapping = {}
#     dataset_subject_criteria = {}

#     for key, mapping in metadata_mapping.items():
#         if key in inclusion_criteria and key not in ['dataset_id', 'subject_id']:
#             # If the key has a specific list of allowed values, filter the mapping
#             allowed_values = inclusion_criteria[key]
#             if allowed_values is not None:  # Check if specific values are provided
#                 limited_mapping = {
#                     value: idx
#                     for value, idx in mapping.items()
#                     if value in allowed_values
#                 }
#                 allowed_metadata_mapping[key] = limited_mapping
#             else:
#                 # If no specific values are provided (None), copy the mapping as is
#                 allowed_metadata_mapping[key] = mapping
#         elif key in ['dataset_id', 'subject_id']:
#             # Handle dataset_id and subject_id separately
#             dataset_subject_criteria[key] = inclusion_criteria[key]
#         else:
#             # If the key is not in the inclusion_criteria, copy the mapping as is
#             allowed_metadata_mapping[key] = mapping

#     return allowed_metadata_mapping, dataset_subject_criteria

# def apply_inclusion_criteria(metadata_mapping, inclusion_criteria):
#     """
#     Allows the metadata mapping based on the allowed values and maximum number of subjects.

#     Parameters:
#     - metadata_mapping (dict): The original metadata mapping.
#     - inclusion_criteria (dict): A dictionary specifying allowed values for each metadata field.

#     Returns:
#     - allowed_metadata_mapping (dict): The limited metadata mapping.
#     """
#     allowed_metadata_mapping = {}
#     for key, mapping in metadata_mapping.items():
#         if key in inclusion_criteria:
#             # If the key has a specific list of allowed values, filter the mapping
#             allowed_values = inclusion_criteria[key]
#             if (
#                 allowed_values is not None
#             ):  # Check if specific values are provided
#                 limited_mapping = {
#                     value: idx
#                     for value, idx in mapping.items()
#                     if value in allowed_values
#                 }
#                 allowed_metadata_mapping[key] = limited_mapping
#             else:
#                 # If no specific values are provided (None), copy the mapping as is
#                 allowed_metadata_mapping[key] = mapping
#         else:
#             # If the key is not in the inclusion_criteria, copy the mapping as is
#             allowed_metadata_mapping[key] = mapping

#     return allowed_metadata_mapping


# def apply_inclusion_criteria(metadata_mapping, allow_dict):
#     """
#     Allows the metadata mapping based on the allowed values and maximum number of subjects.

#     Parameters:
#     - metadata_mapping (dict): The original metadata mapping.
#     - allow_dict (dict): A dictionary specifying allowed values for each metadata field.

#     Returns:
#     - allowed_metadata_mapping (dict): The limited metadata mapping.
#     """
#     allowed_metadata_mapping = {}
#     for key, mapping in metadata_mapping.items():
#         if key in allow_dict:
#             # Filter the mapping based on the allowed values
#             allowed_values = allow_dict[key]
#             limited_mapping = {
#                 value: idx
#                 for value, idx in mapping.items()
#                 if value in allowed_values
#             }
#             allowed_metadata_mapping[key] = limited_mapping
#         else:
#             # If the key is not in the allow_dict, copy the mapping as is
#             allowed_metadata_mapping[key] = mapping

#     return allowed_metadata_mapping


# def apply_sliding_window_data_augmentation(segment_tensor):
#     WINDOW_SIZE_PTS = int(
#         CONFIG["WINDOW_SIZE_SEC"] * CONFIG["TGT_FS"] / CONFIG["OVERLAP_FACTOR"]
#     )
#     start = random.randint(0, segment_tensor.shape[-1] - WINDOW_SIZE_PTS)
#     end = start + WINDOW_SIZE_PTS
#     segment_tensor = segment_tensor[..., start:end]
#     return segment_tensor
