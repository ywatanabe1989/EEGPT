#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-01-26 11:19:09 (ywatanabe)"

import os
import random
import sys

import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
import torch
from models.EEGPT import EEGPT
from scripts.utils.SQLiteDataset import SQLiteDataset
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

# Configurations
CONFIG, sys.stdout, sys.stderr, plt = mngs.gen.start(
    sys,
    plt,
    os=os,
    random=random,
    torch=torch,
    np=np,
)

DEBUG = False
if DEBUG:
    CONFIG["N_EPOCHS"] = 2
    CONFIG["BATCH_SIZE"] = 64

WINDOW_SIZE_PTS = CONFIG["WINDOW_SIZE_SEC"] * CONFIG["TGT_FS"]


# Functions
def base_step(model, i_batch, batch, i_global, metrics, step="Training"):
    if step == "Training":
        model.train()
        optimizer.zero_grad()  # Zero the gradients before running the backward pass.
    else:
        model.eval()

    Db, Sb, Xb = batch  # Dataset ID, Subject ID, EEG signal

    # Sliding window data augumentation
    start = random.randint(0, Xb.shape[-1] - WINDOW_SIZE_PTS)
    end = start + WINDOW_SIZE_PTS
    Xb = Xb[..., start:end]

    # Normalization
    Xb = mngs.gen.to_z(Xb, axis=2)
    Xb = mngs.gen.to_z(Xb, axis=1)
    Xb = Xb.float().cuda()

    loss = model(Xb)

    if step == "Training":
        loss.backward()  # Backpropagation
        optimizer.step()  # Update the weights

    # Buffering
    loss = loss.detach().cpu().item()
    metrics["step"].append(step)
    metrics["i_global"].append(i_global)
    metrics["i_epoch"].append(i_epoch)
    metrics["i_batch"].append(i_batch)
    metrics["loss"].append(loss)

    if i_batch % 10 == 0:
        print(
            f"{step}, Epoch# {i_epoch}, Batch #{i_batch}, Loss: {round(loss, 8)}"
        )

    return metrics


# Loads
sqlite_dataset = SQLiteDataset(db_path=CONFIG["DB_PATH"], show=True)

# Data splitting
ratio_tra = 0.6
ratio_val = 0.2
ratio_tes = 0.2
ds_tra, ds_val, ds_tes = random_split(
    sqlite_dataset, [ratio_tra, ratio_val, ratio_tes]
)

dl_tra = DataLoader(
    ds_tra,
    batch_size=CONFIG["BATCH_SIZE"],
    shuffle=True,
    num_workers=10,
    drop_last=True,
)

dl_val = DataLoader(
    ds_val,
    batch_size=CONFIG["BATCH_SIZE"],
    shuffle=True,
    num_workers=10,
    drop_last=False,
)

dl_tes = DataLoader(
    ds_tes,
    batch_size=CONFIG["BATCH_SIZE"],
    shuffle=True,
    num_workers=10,
    drop_last=False,
)

# Model
model = EEGPT().cuda()
optimizer = SGD(model.parameters(), lr=float(CONFIG["LEARNING_RATE"]))

# Main
metrics = mngs.gen.listed_dict()
i_global = 0
for i_epoch, epoch in tqdm(enumerate(range(CONFIG["N_EPOCHS"]))):

    # Training
    print()
    for i_batch, batch in enumerate(dl_tra):
        if DEBUG and (i_batch > 30):
            continue
        base_step(model, i_batch, batch, i_global, metrics, step="Training")
        i_global += 1

    # Validating
    print()
    for i_batch, batch in enumerate(dl_val):
        if DEBUG and (i_batch > 30):
            continue

        base_step(model, i_batch, batch, i_global, metrics, step="Validation")

# Testing
print()
for i_batch, batch in enumerate(dl_tes):
    if DEBUG and (i_batch > 30):
        continue
    base_step(model, i_batch, batch, i_global, metrics, step="Test")

# Closes the database
sqlite_dataset.close()

# Saves
mngs.io.save(model, CONFIG["SDIR"] + "checkpoint.pth")
mngs.io.save(pd.DataFrame(metrics), CONFIG["SDIR"] + "metrics.csv")

# Closes
mngs.gen.close(CONFIG)
