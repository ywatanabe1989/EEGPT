#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-09 09:07:56 (ywatanabe)"

import os
import random
import sys

import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
import torch
from models.EEGPT import EEGPT
from scripts.data.database.SQLiteDataset import load_ds_dls  # SQLiteDataset
from torch.optim import SGD, Adam

# Configurations
CONFIG, sys.stdout, sys.stderr, plt, cc = mngs.gen.start(
    sys,
    plt,
    os=os,
    random=random,
    torch=torch,
    np=np,
)
WINDOW_SIZE_PTS = CONFIG["WINDOW_SIZE_SEC"] * CONFIG["TGT_FS"]


# Functions
def base_step(
    model,
    optimizer,
    batch,
    i_epoch,
    i_batch,
    i_global,
    metrics,
    step="Training",
):
    if CONFIG["IS_DEBUG"] and (i_batch > 50):
        return None

    # Model mode
    if step == "Training":
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    # Batched Dataset ID, Subject ID, and EEG signal
    Db, Sb, Xb = batch

    # Sliding window as a data augumentation method
    start = random.randint(0, Xb.shape[-1] - WINDOW_SIZE_PTS)
    end = start + WINDOW_SIZE_PTS
    Xb = Xb[..., start:end]

    # Normalization after cropping
    Xb = mngs.gen.to_z(Xb, axis=2)
    Xb = mngs.gen.to_z(Xb, axis=1)
    Xb = Xb.float().cuda()

    # Remove samples with nan values
    Xb = Xb[~Xb.isnan().any(-1).any(-1)]

    # Forward pass
    loss = model(Xb)

    # Backpropagation and weight update
    if step == "Training":
        loss.backward()
        optimizer.step()

    # Buffering
    loss = loss.detach().cpu().item()
    metrics["step"].append(step)
    metrics["i_global"].append(i_global)
    metrics["i_epoch"].append(i_epoch)
    metrics["i_batch"].append(i_batch)
    metrics["loss"].append(loss)

    # Print
    if i_batch == 0:
        print(f"\n{'-'*40}\n")
    if i_batch % 10 == 0:
        print(
            f"\n{step}, Epoch# {i_epoch}, Batch #{i_batch}, Loss: {round(loss, 8)}"
        )

    return metrics


def train_and_eval(dls):
    # Model
    model = EEGPT().cuda()
    optimizer = SGD(model.parameters(), lr=float(CONFIG["LEARNING_RATE"]))

    # Main
    metrics = mngs.gen.listed_dict()
    i_global = 0
    for i_epoch, epoch in enumerate(range(CONFIG["N_EPOCHS"])):

        # Training
        for i_batch, batch in enumerate(dls["tra"]):
            base_step(
                model,
                optimizer,
                batch,
                i_epoch,
                i_batch,
                i_global,
                metrics,
                step="Training",
            )
            i_global += 1

        # Validating
        for i_batch, batch in enumerate(dls["val"]):
            base_step(
                model,
                optimizer,
                batch,
                i_epoch,
                i_batch,
                i_global,
                metrics,
                step="Validation",
            )

    # Testing
    for i_batch, batch in enumerate(dls["tes"]):
        base_step(
            model,
            optimizer,
            batch,
            i_epoch,
            i_batch,
            i_global,
            metrics,
            step="Test",
        )

    return model, pd.DataFrame(metrics)


def main():
    # Loads
    ds, dls = load_ds_dls("./config/Pretrain/OpenNeuro.yaml")

    # Main
    model, metrics_df = train_and_eval(dls)
    fig = mngs.ml.plt.learning_curve(metrics_df, yscale="log")
    # *** ValueError: Source ('src') and target ('tgt') must have the same number of elements.

    # Saves
    mngs.io.save(model, CONFIG["SDIR"] + "checkpoint.pth")
    mngs.io.save(metrics_df, CONFIG["SDIR"] + "metrics.csv")
    mngs.io.save(fig, CONFIG["SDIR"] + "learning_curve.png")

    # Closes
    mngs.gen.close(CONFIG)


if __name__ == "__main__":
    main()
    # print(metrics_df)
    #          step  i_global  i_epoch  i_batch      loss
    # 0    Training         0        0        0  0.721624
    # 1    Training         1        0        1  0.715103
    # 2    Training         2        0        2  0.709523
    # 3    Training         3        0        3  0.721214
    # 4    Training         4        0        4  0.731842
    # ..        ...       ...      ...      ...       ...
    # 250      Test      5956        1       46  0.739182
    # 251      Test      5956        1       47  0.742658
    # 252      Test      5956        1       48  0.720806
    # 253      Test      5956        1       49  0.740011
    # 254      Test      5956        1       50  0.720261

    #   [255 rows x 5 columns]
