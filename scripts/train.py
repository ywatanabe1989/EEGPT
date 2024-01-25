#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-01-24 14:33:05 (ywatanabe)"

import random

import mngs
from scripts.utils.SQLiteDataset import SQLiteDataset
from torch.utils.data import DataLoader, Dataset

CONFIG = mngs.io.load("./config/global.yaml")

WINDOW_SIZE_PTS = CONFIG["WINDOW_SIZE_SEC"] * CONFIG["TGT_FS"]
sqlite_dataset = SQLiteDataset(CONFIG["DB_PATH"])

data_loader = DataLoader(
    sqlite_dataset,
    batch_size=CONFIG["BATCH_SIZE"],
    shuffle=True,
    num_workers=16,
    drop_last=True,
)

for batch in data_loader:
    dataset_ids, subject_ids, X = batch
    X = mngs.ml.utils.sliding_window_data_augmentation(
        X, WINDOW_SIZE_PTS
    )  # torch.Size([64, 14, 256])
