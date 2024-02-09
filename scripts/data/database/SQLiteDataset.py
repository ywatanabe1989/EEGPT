#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-09 18:48:26 (ywatanabe)"

import contextlib  # Add this line to import contextlib
import json
import random
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
from scripts.data.database.check_db import check_db
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy("file_system")

CONFIG = mngs.gen.load_configs(show=False)


def adjust_splits_to_meet_criteria(df, split_criteria, split_ratios):
    """
    Adjust the data splits to ensure they meet the specified split criteria, handling empty criteria.

    Parameters:
    - df (pd.DataFrame): Combined dataset including potential training, validation, and test sets.
    - split_criteria (list): List of criteria for ensuring unique splits; can be empty.
    - split_ratios (dict): Ratios for splitting datasets into training, validation, and test sets.

    Returns:
    - df_tra (pd.DataFrame): Adjusted training dataset.
    - df_val (pd.DataFrame): Adjusted validation dataset.
    - df_tes (pd.DataFrame): Adjusted test dataset.
    """
    if split_criteria:
        # Create a unique key for each combination of split criteria if criteria are provided
        df["unique_key"] = df.apply(
            lambda row: "_".join(
                [str(row[crit]) for crit in split_criteria if crit in row]
            ),
            axis=1,
        )
        unique_keys = df["unique_key"].unique()
    else:
        # Use row indices as unique keys if no criteria are provided
        unique_keys = df.index.to_numpy()

    np.random.shuffle(unique_keys)  # Randomize order of keys

    # Determine split indices based on the provided ratios
    total_keys = len(unique_keys)
    tra_end = int(total_keys * split_ratios["tra"])
    val_end = tra_end + int(total_keys * split_ratios["val"])

    # Split the unique keys according to the calculated indices
    tra_keys, val_keys, tes_keys = (
        unique_keys[:tra_end],
        unique_keys[tra_end:val_end],
        unique_keys[val_end:],
    )

    if split_criteria:
        # Filter the DataFrame based on these unique keys to create splits
        df_tra = df[df["unique_key"].isin(tra_keys)]
        df_val = df[df["unique_key"].isin(val_keys)]
        df_tes = df[df["unique_key"].isin(tes_keys)]

        # Drop the temporary 'unique_key' column
        df_tra.drop(columns=["unique_key"], inplace=True)
        df_val.drop(columns=["unique_key"], inplace=True)
        df_tes.drop(columns=["unique_key"], inplace=True)
    else:
        # Directly split the DataFrame based on indices if no criteria are provided
        df_tra = df.loc[tra_keys]
        df_val = df.loc[val_keys]
        df_tes = df.loc[tes_keys]

    return df_tra, df_val, df_tes


class SQLiteDataset(Dataset):
    """
    Example:
        # Construct dataset
        LPATH_DB = "./data/Dementia/dbs/eeg_data.db"

        INCLUSION_CRITERIA_TRA = {
            "disease_type": ["AD", "DLB", "NPH", "HV"],
            "cognitive_level": ["Normal", "Dementia"],
            "institution": ["Osaka University"],
        }
        INCLUSION_CRITERIA_VAL = {
            "disease_type": ["AD", "DLB", "NPH", "HV"],
            "cognitive_level": ["Normal", "Dementia"],
            "institution": ["Osaka University"],
        }
        INCLUSION_CRITERIA_TES = {
            "disease_type": ["AD", "DLB", "NPH", "HV"],
            "cognitive_level": ["Normal", "Dementia"],
            "institution": ["Nippon Life Hospital", "Kochi University"],
        }
        SPLIT_CRITERIA_TRA_VAL = ["subject_id"]
        SPLIT_CRITERIA_VAL_TES = [
            "subject_id",
            "institution",
        ]
        SPLIT_RATIOS = {"tra": 0.6, "val": 0.2, "tes": 0.2}
        BALANCING_CRITERIA_TRA = ["disease_type"]

        ds = SQLiteDataset(
            lpath_db=LPATH_DB,
            inclusion_criteria_tra=INCLUSION_CRITERIA_TRA,
            inclusion_criteria_val=INCLUSION_CRITERIA_VAL,
            inclusion_criteria_tes=INCLUSION_CRITERIA_TES,
            split_criteria_tra_val=SPLIT_CRITERIA_TRA_VAL,
            split_criteria_val_tes=SPLIT_CRITERIA_VAL_TES,
            split_ratios=RATIOS,
            balancing_criteria_tra=BALANCING_CRITERIA_TRA,
            show=False,
        )
        dataloaders = ds.to_dataloaders()
    """

    def __init__(
        self,
        lpath_db,
        meta_all=None,
        inclusion_criteria_tra={},
        inclusion_criteria_val={},
        inclusion_criteria_tes={},
        split_criteria_tra_val=[],
        split_criteria_val_tes=[],
        split_ratios={"tra": 0.6, "val": 0.2, "tes": 0.2},
        balancing_criteria_tra=[],
        apply_sliding_window=True,
        verbose=False,
    ):
        # Attributes
        self.lpath_db = lpath_db
        self.df = self.load_meta_as_df(self.lpath_db)
        self.meta_all = self.gen_meta_all()
        self.inclusion_criteria_tra = inclusion_criteria_tra
        self.inclusion_criteria_val = inclusion_criteria_val
        self.inclusion_criteria_tes = inclusion_criteria_tes
        self.split_criteria_tra_val = split_criteria_tra_val
        self.split_criteria_val_tes = split_criteria_val_tes
        self.balancing_criteria_tra = balancing_criteria_tra
        self.split_ratios = split_ratios
        self.apply_sliding_window = apply_sliding_window

        # Combines criteria
        self.inclusion_criteria_combined = combine_inclusion_criteria(
            self.inclusion_criteria_tra,
            self.inclusion_criteria_val,
            self.inclusion_criteria_tes,
        )

        # Maps variables to int representations
        self.meta_map = self.gen_meta_included(
            self.meta_all, self.inclusion_criteria_combined
        )

        # Determines indices according to various criteria
        self.indi_tra, self.indi_val, self.indi_tes = get_indi(
            self.df,
            CONFIG,
            self.inclusion_criteria_tra,
            self.inclusion_criteria_val,
            self.inclusion_criteria_tes,
            self.balancing_criteria_tra,
            self.split_criteria_tra_val,
            self.split_criteria_val_tes,
            self.split_ratios,
        )
        self.allowed_indices = self.indi_tra + self.indi_val + self.indi_tes

        # Split dataset
        self.datasets = {"tra": [], "val": [], "tes": []}
        self.split_dataset()

        self.length = self.__len__()

        if verbose:
            print(
                f"\nThe metadata of the entire database, {self.lpath_db}, is as follows:\n{self.df}"
            )
            print(
                f"\nIn the current analysis, metadata is mapped as follows:\n"
            )
            pprint(self.meta_map)

    @classmethod
    def load_meta_as_df(cls, lpath_db):
        """
        Load data from the SQLite database as a pandas DataFrame.

        Parameters:
        - lpath_db (str): The file path to the SQLite database.

        Returns:
        - df (pandas.DataFrame): DataFrame containing data.
        """
        print("\nLoading metadata as a DataFrame...")
        if CONFIG["IS_DEBUG"]:
            # print("Debug mode: Loading a random subset of rows.")
            # Modify the query to load a random subset of 100 rows
            N_ROWS = int(
                CONFIG["N_LPATHS_TO_LOAD"] * CONFIG["MAX_N_WINDOWS_PER_SUB"]
            )
            with cls.get_cursor(lpath_db) as cursor:
                cursor.execute(
                    f"SELECT id, dataset_id, subject_id, metadata FROM eeg_data ORDER BY RANDOM() LIMIT {N_ROWS}"
                )
        with cls.get_cursor(lpath_db) as cursor:
            cursor.execute(
                "SELECT id, dataset_id, subject_id, metadata FROM eeg_data"
            )
            rows = cursor.fetchall()
        data_dicts = []
        for row in rows:
            data_dict = json.loads(row[3])
            data_dict["id"] = row[0]
            data_dict["dataset_id"] = row[1]
            data_dict["subject_id"] = row[2]
            data_dicts.append(data_dict)
        df = pd.DataFrame(data_dicts).set_index("id")
        return df

    def gen_meta_all(
        self,
    ):
        meta_all = OrderedDict()
        for col in self.df.columns:
            if not col in ["dataset_id", "subject_id"]:
                _meta_dict = OrderedDict(
                    {key: ii for ii, key in enumerate(self.df[col].unique())}
                )
                meta_all[col] = _meta_dict
        return meta_all

    # @classmethod
    # def gen_meta_all(cls, lpath_db):
    #     """
    #     Generates independent mappings for each metadata tag from a SQLite database.

    #     Parameters:
    #     - lpath_db (str): The file path to the SQLite database.

    #     Returns:
    #     - metadata_mapping (OrderedDict of OrderedDict): An ordered dictionary containing mappings for each metadata tag.
    #     """
    #     print(f"\nGenerating metadata mapping...\n")

    #     # Fetch unique metadata values
    #     conn = sqlite3.connect(lpath_db)
    #     cursor = conn.cursor()
    #     cursor.execute("SELECT DISTINCT metadata FROM eeg_data")
    #     unique_metadata_values = cursor.fetchall()
    #     conn.close()

    #     # Initialize an OrderedDict to collect values for each metadata tag
    #     metadata_values = OrderedDict()

    #     # Iterate over unique metadata values and populate the metadata_values dictionary
    #     for (json_string,) in unique_metadata_values:
    #         metadata = json.loads(json_string)
    #         for key in metadata:
    #             if key not in metadata_values:
    #                 metadata_values[key] = set()
    #             metadata_values[key].add(metadata[key])

    #     # Create independent mappings for each metadata tag using OrderedDict
    #     metadata_mapping = OrderedDict()
    #     for key, values in metadata_values.items():
    #         metadata_mapping[key] = OrderedDict(
    #             (value, index) for index, value in enumerate(sorted(values))
    #         )

    #     # Print the metadata mapping
    #     print("\nMetadata mapping:")
    #     for metadata_key, mapping in metadata_mapping.items():
    #         print(f"\n{metadata_key.capitalize()}:")
    #         for value, index in mapping.items():
    #             print(f"  (ID: {index}) {value}")

    #     return metadata_mapping

    @staticmethod
    def gen_meta_included(metadata_mapping, allow_dict):
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
                # Reallocate values from zero
                limited_mapping = {
                    value: ii
                    for ii, (value, idx) in enumerate(limited_mapping.items())
                }
                allowed_metadata_mapping[key] = limited_mapping
            else:
                # If the key is not in the allow_dict, copy the mapping as is
                allowed_metadata_mapping[key] = mapping

        return allowed_metadata_mapping

    @staticmethod
    @contextlib.contextmanager
    def get_cursor(lpath_db):
        conn = sqlite3.connect(lpath_db)
        try:
            yield conn.cursor()
        except sqlite3.DatabaseError as e:
            print(f"An error occurred: {e}")
        finally:
            conn.close()

    def __getitem__(self, index):
        index = self.allowed_indices[index]

        # The index here is a direct index into the dataset.
        query = "SELECT dataset_id, subject_id, segment, metadata FROM eeg_data WHERE id = ?"
        # params = [index + 1]  # Assuming that the database IDs start at 1
        params = [index]  # Assuming that the database IDs start at 1

        with self.get_cursor(self.lpath_db) as cursor:
            cursor.execute(query, params)
            row = cursor.fetchone()

        if not row:
            raise IndexError(f"Index {index} does not have data.")

        dataset_id, subject_id, segment, metadata_json = row
        segment_tensor = torch.from_numpy(
            np.frombuffer(segment, dtype=np.float64)
            .astype(np.float32)
            .reshape(len(CONFIG["TGT_TRANVERSE_BIPOLAR_MONTAGE"]), -1)
            .copy()
        ).float()
        metadata_tensor = self._meta2tensor(metadata_json)
        return dataset_id, subject_id, segment_tensor, metadata_tensor
        # return dataset_id, subject_id, segment_tensor, metadata_json

    def __len__(self):
        return len(self.allowed_indices)

    def _meta2tensor(self, metadata_json):
        metadata_dict = json.loads(metadata_json)
        encoded_features = []
        for key, value in metadata_dict.items():
            encoded_value = self.meta_map[key][value]
            encoded_features.append(encoded_value)
        return torch.tensor(encoded_features, dtype=torch.long)

    def _tensor2meta(self, tensor):
        reverse_meta_map = {
            key: {v: k for k, v in value_map.items()}
            for key, value_map in self.meta_map.items()
        }
        indices = tensor.tolist()
        metadata_dict = {}
        for key, index in zip(self.meta_map.keys(), indices):
            metadata_value = reverse_meta_map[key][index]
            metadata_dict[key] = metadata_value
        metadata_json = json.dumps(metadata_dict)
        return metadata_json

    def decode_meta(self, MM):
        meta_dicts = [json.loads(self._tensor2meta(row)) for row in MM]
        meta_srs = [
            pd.Series(data=md.values(), index=md.keys()) for md in meta_dicts
        ]
        meta_df = pd.concat(meta_srs, axis=1).T
        return meta_df

    def split_dataset(
        self,
    ):
        print("\nSplitting the dataset for training, validation, and test...")

        # Revert index of each dataset
        revert_dict = {
            ii: i_ii for i_ii, ii in enumerate(self.allowed_indices)
        }
        indi_tra_cvted = [revert_dict[ii] for ii in self.indi_tra]
        indi_val_cvted = [revert_dict[ii] for ii in self.indi_val]
        indi_tes_cvted = [revert_dict[ii] for ii in self.indi_tes]

        # Convert index lists to Subset objects
        self.datasets["tra"] = torch.utils.data.Subset(self, indi_tra_cvted)
        self.datasets["val"] = torch.utils.data.Subset(self, indi_val_cvted)
        self.datasets["tes"] = torch.utils.data.Subset(self, indi_tes_cvted)

        # Optionally, log subset sizes for verification
        print(f"\nTraining set size: {len(self.datasets['tra']):,}")
        print(f"\nValidation set size: {len(self.datasets['val']):,}")
        print(f"\nTest set size: {len(self.datasets['tes']):,}")

    def to_dataloaders(self, batch_size=64, num_workers=1, shuffle=True):
        # Create DataLoader instances for the dataset splits
        dataloaders = {}
        for split in ["tra", "val", "tes"]:
            try:
                dataloaders[split] = DataLoader(
                    self.datasets[split],
                    batch_size=batch_size,
                    shuffle=shuffle if split == "tra" else False,
                    num_workers=num_workers,
                    collate_fn=custom_collate_fn
                    if self.apply_sliding_window
                    else None,
                )
            except Exception as e:
                print(e)
        return dataloaders


def apply_sliding_window_data_augmentation(segment_tensor):
    WINDOW_SIZE_PTS = int(CONFIG["WINDOW_SIZE_SEC"] * CONFIG["TGT_FS"])
    start = random.randint(0, segment_tensor.shape[-1] - WINDOW_SIZE_PTS)
    end = start + WINDOW_SIZE_PTS
    segment_tensor = segment_tensor[..., start:end]
    return segment_tensor


# Define the custom collate function
def custom_collate_fn(batch):
    augmented_batch = []
    for sample in batch:
        dataset_id, subject_id, segment_tensor, metadata_tensor = sample
        # Apply sliding window data augmentation to the segment_tensor
        augmented_segment_tensor = apply_sliding_window_data_augmentation(
            segment_tensor
        )
        augmented_sample = (
            dataset_id,
            subject_id,
            augmented_segment_tensor,
            metadata_tensor,
        )
        augmented_batch.append(augmented_sample)
    # Use the default collate function to merge the augmented samples into a batch
    return torch.utils.data.dataloader.default_collate(augmented_batch)


def combine_inclusion_criteria(*criteria):
    combined_criteria = {}

    # Iterate over all dictionaries passed to the function
    for criterion in criteria:
        # Iterate over key-value pairs in each dictionary
        for key, values in criterion.items():
            # If the key is already in the combined dictionary, update it with new values
            if key in combined_criteria:
                combined_criteria[key].update(values)
            # If the key is not in the combined dictionary, add it
            else:
                combined_criteria[key] = set(values)

    # Convert sets back to lists for the final output
    for key in combined_criteria:
        combined_criteria[key] = list(combined_criteria[key])

    return combined_criteria


def get_indi(
    df,
    CONFIG,
    INCLUSION_CRITERIA_TRA,
    INCLUSION_CRITERIA_VAL,
    INCLUSION_CRITERIA_TES,
    BALANCING_CRITERIA_TRA,
    SPLIT_CRITERIA_TRA_VAL,
    SPLIT_CRITERIA_VAL_TES,
    split_ratios={"tra": 0.6, "val": 0.2, "tes": 0.2},
):
    """
    1. Define the data to be used according to the INCLUSION_CRITERIA.
    2. Limit the number of samples from each subject to a maximum of CONFIG["MAX_N_WINDOWS_PER_SUB"] to prevent sampling bias.
    3. Divide the data into training, validation, and test sets based on the respective CRITERIA, disregarding split_ratios.
    4. Verify that the data splits adhere to the SPLIT_CRITERIA.
       4-1. If not, adjust the splits to comply with the SPLIT_CRITERIA, using split_ratios if necessary.
    5. Balance the training dataset to prevent imbalances.
    6. Identify the indices of datasets
    7. Confirm the critearia are met.
    """

    def filter_by_criteria(df, criteria):
        mask = pd.Series(True, index=df.index)
        for key, values in criteria.items():
            mask &= df[key].isin(values)
        return df[mask]

    def limit_segments_per_sub(df):
        def _limit_segments_per_sub(group):
            return group.sample(
                n=min(len(group), CONFIG["MAX_N_WINDOWS_PER_SUB"]),
                random_state=42,
            )

        df_reset = df.reset_index()
        df_limited = df_reset.groupby("subject_id").apply(
            _limit_segments_per_sub
        )
        df_limited = df_limited.set_index("id")
        return df_limited

    def balance_dataset(df, criteria):
        if criteria:
            min_count = df[criteria[0]].value_counts().min()
            return df.groupby(criteria[0], group_keys=False).apply(
                lambda x: x.sample(min_count)
            )
        else:
            return df

    def check_split_criteria(df1, df2, split_criteria):
        # Check if the split criteria columns are in both dataframes
        for criterion in split_criteria:
            if criterion not in df1.columns or criterion not in df2.columns:
                print(
                    f"Split criteria '{criterion}' not found in both dataframes."
                )
                return False

            # Check if the unique values of the split criteria are mutually exclusive
            train_val_unique_values = set(df1[criterion].unique())
            test_unique_values = set(df2[criterion].unique())
            if not train_val_unique_values.isdisjoint(test_unique_values):
                print(
                    f"Split criteria '{criterion}' values are not mutually exclusive."
                )
                return False

        # If all checks pass, the split meets the criteria
        return True

    def split_data(df, split_criteria, split_size=0.2, random_state=42):
        valid_split_criteria = list(
            df.columns[df.columns.isin(split_criteria)]
        )
        if not valid_split_criteria:
            raise ValueError(
                f"Split creteria '{split_criteria}' not found in the DataFrame."
            )

        if split_size == 0:
            return df, None

        merged_col = mngs.ml.utils.merge_labels(
            df[valid_split_criteria].squeeze()
        )
        merged_uq = np.unique(merged_col)

        # Split the unique combinations into training/validation and test sets
        uq_1, uq_2 = train_test_split(
            merged_uq, test_size=split_size, random_state=random_state
        )

        # df_1 = df[mngs.gen.search(uq_1, merged_col, as_bool=True)[0]]
        # df_2 = df[mngs.gen.search(uq_2, merged_col, as_bool=True)[0]]

        df_1 = df[merged_col.isin(uq_1)]
        df_2 = df[merged_col.isin(uq_2)]

        for col in valid_split_criteria:
            assert set(df_1[col]).isdisjoint(set(df_2[col]))

        return df_1, df_2

    # 1. Define the data to be used according to the INCLUSION_CRITERIA.
    combined_inclusion_criteria = combine_inclusion_criteria(
        INCLUSION_CRITERIA_TRA,
        INCLUSION_CRITERIA_VAL,
        INCLUSION_CRITERIA_TES,
    )
    df = filter_by_criteria(df, combined_inclusion_criteria)

    # 2. Limit the number of samples from each subject to a maximum of CONFIG["MAX_N_WINDOWS_PER_SUB"] to prevent sampling bias.
    df = limit_segments_per_sub(df)

    # 3. Divide the data into training, validation, and test sets based on the respective CRITERIA, disregarding split_ratios. Note that, exclusivenes is not considered yet.
    df_tra = filter_by_criteria(df, INCLUSION_CRITERIA_TRA)
    df_val = filter_by_criteria(df, INCLUSION_CRITERIA_VAL)
    df_tes = filter_by_criteria(df, INCLUSION_CRITERIA_TES)

    # 4. Verify that the data splits adhere to the SPLIT_CRITERIA.
    #    4-1. If not, adjust the splits to comply with the SPLIT_CRITERIA, using split_ratios if necessary.
    if not check_split_criteria(df_tra, df_tes, SPLIT_CRITERIA_TRA_VAL):
        # Combine training and validation data temporarily for splitting
        df_tra_val_tes = (
            pd.concat([df_tra, df_val, df_tes])
            .reset_index()
            .drop_duplicates()
            .set_index("id")
        )

        # Further split the training/validation set into training and validation sets
        df_tra_val, df_tes = split_data(
            df_tra_val_tes,
            SPLIT_CRITERIA_VAL_TES,
            split_size=split_ratios["tes"],
        )

    if not check_split_criteria(df_tra, df_val, SPLIT_CRITERIA_TRA_VAL):
        # Combine training and validation data temporarily for splitting
        df_tra_val = (
            pd.concat([df_tra, df_val])
            .reset_index()
            .drop_duplicates()
            .set_index("id")
        )

        # Further split the training/validation set into training and validation sets
        df_tra, df_val = split_data(
            df_tra_val,
            SPLIT_CRITERIA_TRA_VAL,
            split_size=split_ratios["val"]
            / (split_ratios["tra"] + split_ratios["val"]),
        )

    # 5. Balance the training dataset to prevent imbalances.
    df_tra = balance_dataset(df_tra, BALANCING_CRITERIA_TRA)

    # 6. Identify the indices of datasets
    tra_indices = df_tra.reset_index()["id"] if df_tra is not None else [None]
    val_indices = df_val.reset_index()["id"] if df_val is not None else [None]
    tes_indices = df_tes.reset_index()["id"] if df_tes is not None else [None]

    # 7. Confirm the critearia are met.
    # Confirmation
    assert set(tra_indices).isdisjoint(set(val_indices))
    assert set(val_indices).isdisjoint(set(tes_indices))
    assert set(tes_indices).isdisjoint(set(tra_indices))

    if not check_split_criteria(df_tra, df_val, SPLIT_CRITERIA_TRA_VAL):
        raise ValueError(
            "Validation data is not exclusive from training dataset"
        )

    if not check_split_criteria(df_tra, df_tes, SPLIT_CRITERIA_VAL_TES):
        raise ValueError(
            "Test data is not exclusive from training and validation datasets"
        )

    return list(tra_indices), list(val_indices), list(tes_indices)


def load_ds_dls(lpath_ds_config, **kwargs):
    ds_config = mngs.io.load(lpath_ds_config, lower=True)
    # ds_config_lower = {key.lower(): value for key, value in ds_config.items()}
    ds = SQLiteDataset(**ds_config, **kwargs)
    dls = ds.to_dataloaders()
    return ds, dls


if __name__ == "__main__":
    import os
    import random
    import sys
    from datetime import datetime

    import matplotlib.pyplot as plt
    import mngs

    CONFIG, sys.stdout, sys.stderr, plt, cc = mngs.gen.start(sys, plt)
    WINDOW_SIZE_PTS = CONFIG["WINDOW_SIZE_SEC"] * CONFIG["TGT_FS"]

    # Construct dataset
    lpath_ds_config = "./config/DownStream/Dementia.yaml"
    ds, dls = load_ds_dls(lpath_ds_config)

    # Main
    for i_epoch in tqdm(range(5)):
        DS_epoch = []
        S_epoch = []
        # M_epoch = []
        DT_epoch = []
        C_epoch = []
        I_epoch = []
        for i_batch, batch in tqdm(enumerate(dls["tra"])):
            """
            batch = next(iter(dls["tra"]))
            batch = next(iter(ds_tra))
            """
            # Batched dataset ids, subject ids, EEG signals, and metadatas
            DSb, Sb, Xb, _Mb = batch

            # Xb = apply_sliding_window_data_augmentation(Xb)
            print(Xb.shape)

            DTb = _Mb[:, 0]  # Disease type
            Cb = _Mb[:, 1]  # Cognitive level
            Ib = _Mb[:, 2]  # Institution

            DS_epoch.append(DSb)
            S_epoch.append(Sb)
            DT_epoch.append(DTb)
            C_epoch.append(Cb)
            I_epoch.append(Ib)
            # M_epoch.append(_Mb)

    ds.decode_meta(_Mb)

    DS_uq = mngs.gen.unique(np.hstack(DS_epoch))
    S_uq = mngs.gen.unique(np.hstack(S_epoch))
    DT_uq = mngs.gen.unique(np.hstack(DT_epoch))
    C_uq = mngs.gen.unique(np.hstack(C_epoch))
    I_uq = mngs.gen.unique(np.hstack(I_epoch))

    print(f"\nLoaded dataset list:\n{DS_uq}")
    print(f"\nLoaded subject list:\n{S_uq}")
    print(f"\nLoaded disease_type list:\n{DT_uq}")
    print(f"\nLoaded cognitive_level list:\n{C_uq}")
    print(f"\nLoaded institution list:\n{I_uq}")

    # END
    mngs.gen.close(CONFIG)

    """
    ./scripts/data/database/SQLiteDataset.py
    """
