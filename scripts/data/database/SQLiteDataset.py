#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-03-12 12:20:26 (ywatanabe)"

import contextlib  # Add this line to import contextlib
import json
import random
import re
import sqlite3
import sys
import warnings
from collections import OrderedDict
from pprint import pprint

import matplotlib.pyplot as plt
import mne
import mngs
import numpy as np
import pandas as pd
import torch
from icecream import ic
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
        batch_size=None,
        from_cache=True,
        verbose=False,
    ):
        # Attributes
        self.ds_config = ds_config

        if batch_size is not None:
            self.ds_config["BATCH_SIZE"] = batch_size

        # Loads metadata
        self.df = self.load_df_from_cache(from_cache)

        self.meta_all = self.gen_meta_all()

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
                f"\nThe metadata of the entire database ({self.ds_config['LPATH_DB']}) is as follows:\n{self.df.columns}"
            )
            print(
                f"\nIn the current analysis, metadata is mapped as follows:\n"
            )
            pprint(self.meta_map)
            print(f"\nTraining + Validation + Test")
            print(f"\ndataset_id:\n{mngs.gen.unique(self.df['dataset_id'])}")
            print(f"\subject_id:\n{mngs.gen.unique(self.df['subject_id'])}")

            for split_str, df in zip(
                ["Training", "Validation", "Test"], [df_tra, df_val, df_tes]
            ):
                print(f"\n{split_str}")
                print(f"\n# of datasets:\n{len(df['dataset_id'].unique())}")
                print(f"\n# of subjects:\n{len(df['subject_id'].unique())}")

    def load_df_from_cache(
        self,
        from_cache=False,
    ):
        cache_path = self.ds_config["LPATH_DB"].replace(
            ".db", "_meta_cache.pkl"
        )

        # Loads from db
        if not from_cache:
            self.df = self.load_meta_as_df(self.ds_config["LPATH_DB"])
            mngs.io.save(self.df, cache_path)

        # Loads from cache pkl
        else:
            try:
                self.df = mngs.io.load(cache_path)
                assert self.df is not None
                print(f"\nLoaded from the cache file: {cache_path}")
            except Exception as e:
                self.df = self.load_meta_as_df(self.ds_config["LPATH_DB"])
                mngs.io.save(self.df, cache_path)

        return self.df

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

        query = "SELECT dataset_id, subject_id, segment, metadata FROM eeg_data WHERE id = ?"
        params = [index]

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
                custom_collate_fn = CustomCollateFnMaker(
                    random_split=True,
                )

                dataloaders[split] = DataLoader(
                    self.datasets[split],
                    batch_size=self.ds_config["BATCH_SIZE"],
                    shuffle=True if split == "tra" else False,
                    num_workers=self.ds_config["NUM_WORKERS"],
                    collate_fn=custom_collate_fn,
                    drop_last=True,
                    worker_init_fn=seed_worker,
                    generator=g,
                )
            except Exception as e:
                print(e)
        return dataloaders


class CustomCollateFnMaker:
    def __init__(self, random_split=False):
        self.random_split = random_split

    @staticmethod
    def sliding_window(Xb, random_split=False):
        WINDOW_SIZE_PTS = int(CONFIG["WINDOW_SIZE_SEC"] * CONFIG["TGT_FS"])
        if Xb.shape[-1] < WINDOW_SIZE_PTS:
            warnings.warn(
                f"The length of EEG data ({Xb.shape[-1]}) is shorter than the window size ({WINDOW_SIZE_PTS}) "
                f"for sliding window data augmentation. NaN data is returned.",
                UserWarning,
            )
            return torch.full(
                (Xb.shape[0], Xb.shape[1], WINDOW_SIZE_PTS), torch.nan
            )
        start = (
            random.randint(0, Xb.shape[-1] - WINDOW_SIZE_PTS)
            if random_split
            else 0
        )
        Xb = Xb[..., start : start + WINDOW_SIZE_PTS]
        return Xb

    # @staticmethod
    # def drop_nan_including_samples(batch):
    #     Xb = batch[2]
    #     indi_nan = Xb.isnan().any(-1).any(-1)
    #     return [item[~indi_nan] for item in batch]

    def __call__(self, batch):
        augmented_batch = []
        for sample in batch:
            dataset_id, subject_id, Xs, metadata_tensor = sample
            Xs = self.sliding_window(Xs, random_split=self.random_split)

            if Xs.isnan().any():
                continue

            augmented_sample = [
                dataset_id,
                subject_id,
                Xs,
                metadata_tensor,
            ]
            augmented_batch.append(augmented_sample)

        return torch.utils.data.dataloader.default_collate(augmented_batch)


def combine_inclusion_criteria(*criteria):
    combined_criteria = {}

    for criterion in criteria:

        if criterion is None:
            continue

        for key, values in criterion.items():
            if key in combined_criteria:
                combined_criteria[key].update(values)
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
    # lpath_ds_config = "./config/DownStream/Dementia.yaml"
    # lpath_ds_config = "./config/Pretrain/OpenNeuro.yaml"
    lpath_ds_config = "./config/DownStream/Epilepsy.yaml"
    ds, dls = load_ds_dls(
        lpath_ds_config, from_cache=False, verbose=True, batch_size=4
    )

    batch = next(iter(dls["tra"]))
    batch = next(iter(dls["val"]))
    batch = next(iter(dls["tes"]))

    # Main
    for i_epoch in tqdm(range(5)):
        D_all = []
        S_all = []
        X_all = []
        M_all = []
        ILAE_all = []
        for i_batch, batch in tqdm(enumerate(dls["val"])):
            # Batched dataset ids, subject ids, EEG signals, and metadatas
            DSb, Sb, Xb, _Mb = batch
            D_all.append(DSb)
            S_all.append(Sb)
            X_all.append(Xb)
            M_all.append(_Mb)
            ILAE_all.append(np.array(ds.decode_meta(_Mb)["ILAE"]).astype(int))

            # Mb_df = ds.decode_meta(_Mb)
    # ds.decode_meta(_Mb)["ILAE"]

    X_all = np.vstack(X_all)
    S_all = np.hstack(S_all)
    M_all = np.vstack(M_all)
    ILAE_all = np.hstack(ILAE_all)

    df = pd.DataFrame(
        columns=["subject", "ILAE"],
        data=np.stack([S_all, ILAE_all]).T,
    )
    df = df.drop_duplicates()
    mngs.gen.uq(M_all)
    ds.decode_meta(_Mb)

    DS_uq = mngs.gen.unique(np.hstack(DS_epoch))
    S_uq = mngs.gen.unique(np.hstack(S_epoch))
    # DT_uq = mngs.gen.unique(np.hstack(DT_epoch))
    # C_uq = mngs.gen.unique(np.hstack(C_epoch))
    # I_uq = mngs.gen.unique(np.hstack(I_epoch))

    print(f"\nLoaded dataset list:\n{DS_uq}")
    print(f"\nLoaded subject list:\n{S_uq}")
    # print(f"\nLoaded disease_type list:\n{DT_uq}")
    # print(f"\nLoaded cognitive_level list:\n{C_uq}")
    # print(f"\nLoaded institution list:\n{I_uq}")

    # END
    mngs.gen.close(CONFIG)

    """
    ./scripts/data/database/SQLiteDataset.py
    """
