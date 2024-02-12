#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-12 22:40:23 (ywatanabe)"

import logging
import os
import random
import sys

import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
import torch
from models.EEGPT import EEGPT
from models.MultiTaskLoss import MultiTaskLoss
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
    Db, Sb, Xb, _Mb = batch

    # Sliding window as a data augumentation method
    start = random.randint(0, Xb.shape[-1] - WINDOW_SIZE_PTS)
    end = start + WINDOW_SIZE_PTS
    Xb = Xb[..., start:end]

    # Normalization after cropping
    Xb = mngs.gen.to_z(Xb, axis=2)
    Xb = mngs.gen.to_z(Xb, axis=1)
    Xb = Xb.float().cuda()

    # Remove samples with nan values
    indi_nan = Xb.isnan().any(-1).any(-1)
    has_nan = indi_nan.sum()

    Xb = Xb[~indi_nan]

    # Forward pass
    loss, bacc = model(Xb)

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
    metrics["bACC"].append(bacc)

    # Print
    if i_batch == 0:
        print(f"\n{'-'*40}\n")

    if i_batch % 100 == 0:
        # print(f"Embedding weights:\n{model.VQ.embedding.weight}")
        print(f"model.VQ.q:\n{model.VQ.q[:5]}")
        print(
            f"argmax:\n{mngs.gen.unique(model.VQ.q.argmax(axis=-1).cpu().detach().numpy())}"
        )
        print(
            f"\n{step}, Epoch# {i_epoch}, Batch #{i_batch}, Loss: {round(loss, 8)}, bACC: {round(bacc, 3)}"
        )

    #     print(
    #         f"\n{step}, Epoch# {i_epoch}, Batch #{i_batch}, Loss: {round(loss, 8)}"
    #     )

    return metrics


def train_and_eval(dls):
    # Model
    model = EEGPT().cuda()

    optimizer = SGD(model.parameters(), lr=float(CONFIG["LEARNING_RATE"]))
    # optimizer = Adam(model.parameters(), lr=float(CONFIG["LEARNING_RATE"]))

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

    # Saves
    mngs.io.save(model, CONFIG["SDIR"] + "checkpoint.pth")
    mngs.io.save(metrics_df, CONFIG["SDIR"] + "metrics.csv")
    mngs.io.save(fig, CONFIG["SDIR"] + "learning_curve.png")

    # Closes
    mngs.gen.close(CONFIG)


if __name__ == "__main__":
    main()
