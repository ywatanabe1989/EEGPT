#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-17 02:08:43 (ywatanabe)"

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
from scripts.data.database.split_datasets import split_datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy("file_system")

CONFIG = mngs.gen.load_configs(show=False)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)


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
        ds_config,
        meta_all=None,
        collate_fn=None,
        verbose=False,
    ):
        # Attributes
        self.ds_config = ds_config

        # Loads metadata
        # self.df = self.load_meta_as_df(self.ds_config["LPATH_DB"])
        cache_path = self.ds_config["LPATH_DB"].replace(
            ".db", "_meta_cache.pkl"
        )
        try:
            self.df = mngs.io.load(cache_path)
            assert self.df is not None
        except Exception as e:
            self.df = self.load_meta_as_df(self.ds_config["LPATH_DB"])
            mngs.io.save(self.df, cache_path)

        self.meta_all = self.gen_meta_all()
        # self.collate_fn = (
        #     sliding_window_collate_fn
        #     if collate_fn is "sliding_window"
        #     else None
        # )

        # Combines criteria
        self.INCLUSION_CRITERIA_COMBINED = combine_inclusion_criteria(
            self.ds_config["INCLUSION_CRITERIA_TRA"],
            self.ds_config["INCLUSION_CRITERIA_VAL"],
            self.ds_config["INCLUSION_CRITERIA_TES"],
        )

        # Maps variables to int representations
        self.meta_map = self.gen_meta_included(
            self.meta_all, self.INCLUSION_CRITERIA_COMBINED
        )

        # Determines indices according to various criteria
        df_tra, df_val, df_tes = split_datasets(self.df, self.ds_config)
        self.indi_tra = list(df_tra.index)
        self.indi_val = list(df_val.index)
        self.indi_tes = list(df_tes.index)
        self.allowed_indices = self.indi_tra + self.indi_val + self.indi_tes

        # Split dataset
        self.datasets = {"tra": [], "val": [], "tes": []}
        self.to_datasets()

        self.length = self.__len__()

        if verbose:
            print(
                f"\nThe metadata of the entire database, {self.ds_config['LPATH_DB']}, is as follows:\n{self.df}"
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

        with self.get_cursor(self.ds_config["LPATH_DB"]) as cursor:
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

    def to_datasets(
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

    def to_dataloaders(
        self,
    ):
        # Create DataLoader instances for the dataset splits
        dataloaders = {}
        for split in ["tra", "val", "tes"]:
            try:
                dataloaders[split] = DataLoader(
                    self.datasets[split],
                    batch_size=self.ds_config["BATCH_SIZE"],
                    shuffle=True if split == "tra" else False,
                    num_workers=self.ds_config["NUM_WORKERS"],
                    collate_fn=sliding_window_collate_fn
                    if split == "tra"
                    else None,
                    drop_last=True,
                    worker_init_fn=seed_worker,
                    generator=g,
                )
            except Exception as e:
                print(e)
        return dataloaders


# Define the custom collate function
def sliding_window(segment_tensor):
    WINDOW_SIZE_PTS = int(CONFIG["WINDOW_SIZE_SEC"] * CONFIG["TGT_FS"])
    start = random.randint(0, segment_tensor.shape[-1] - WINDOW_SIZE_PTS)
    end = start + WINDOW_SIZE_PTS
    segment_tensor = segment_tensor[..., start:end]
    return segment_tensor


def sliding_window_collate_fn(batch):
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


def load_ds_dls(lpath_ds_config, **kwargs):
    ds_config = mngs.io.load(lpath_ds_config)
    ds = SQLiteDataset(ds_config, **kwargs)
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
