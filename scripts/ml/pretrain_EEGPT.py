#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-15 20:24:20 (ywatanabe)"

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
    alpha=1,
    beta=1,
    gamma=1,
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
    # alpha, beta, gamma = 1, 0.3, 0
    commitment_loss, entropy_loss, prediction_loss = model(
        Xb, alpha=alpha, beta=beta, gamma=gamma
    )
    loss = commitment_loss + entropy_loss + prediction_loss

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
        print(
            f"\n{step}, Epoch# {i_epoch}, Batch #{i_batch}, Loss: {round(loss, 8)}"
        )

        print(f"\nprediction loss: {prediction_loss.item():.5f}")
        print(f"\ncommitment loss: {commitment_loss.item():.5f}")
        print(f"\nentropy loss: {entropy_loss.item():.5f}")

        print(
            f"\nmodel.VQ.embedding.weight.data:\n{model.VQ.embedding.weight.data[:5]}"
        )
        print(f"\nmodel.VQ.q:\n{model.VQ.q[:5]}")
        print(f"\nmodel.VQ.q:\n{model.VQ.q[:5]}")

        print(
            f"\nmodel.VQ.q.argmax:\n{mngs.gen.unique(model.VQ.q.argmax(axis=-1).cpu().detach().numpy())}"
        )

    # if i_batch % 100 == 0:
    # print(f"Embedding weights:\n{model.VQ.embedding.weight}")

    #     print(
    #         f"\n{step}, Epoch# {i_epoch}, Batch #{i_batch}, Loss: {round(loss, 8)}"
    #     )

    return metrics


def train_and_eval(dls, alpha=1, beta=1, gamma=1):
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
                alpha=alpha,
                beta=beta,
                gamma=gamma,
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
                alpha=alpha,
                beta=beta,
                gamma=gamma,
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
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

    return model, pd.DataFrame(metrics)


def main(alpha=1.0, beta=1.0, gamma=1.0):
    # Loads
    ds, dls = load_ds_dls("./config/Pretrain/OpenNeuro.yaml")
    len(ds.df["dataset_id"].unique())  # 78
    len(ds.df["subject_id"].unique())  # 6144
    mngs.io.save(ds.df, "./config/Pretrain/OpenNeuro.csv")

    # Main
    model, metrics_df = train_and_eval(
        dls, alpha=alpha, beta=beta, gamma=gamma
    )
    fig = mngs.ml.plt.learning_curve(metrics_df, yscale="log")

    # Saves
    mngs.io.save(model, CONFIG["SDIR"] + "checkpoint.pth")
    mngs.io.save(metrics_df, CONFIG["SDIR"] + "metrics.csv")
    mngs.io.save(fig, CONFIG["SDIR"] + "learning_curve.png")

    # Closes
    mngs.gen.close(CONFIG)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train the EEGPT model with given hyperparameters."
    )
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=1.0,
        help="Weight for the commitment loss",
    )
    parser.add_argument(
        "-b",
        "--beta",
        type=float,
        default=1.0,
        help="Weight for the entropy loss",
    )
    parser.add_argument(
        "-g",
        "--gamma",
        type=float,
        default=1.0,
        help="Weight for the prediction loss",
    )
    args = parser.parse_args()

    print(f"\nargs:\n{args}")

    main(alpha=args.alpha, beta=args.beta, gamma=args.gamma)

    """
    ./scripts/ml/pretrain_EEGPT.py -a 1.0 -b 0.0 -g 0.0
    ./scripts/ml/pretrain_EEGPT.py -a 1.0 -b 0.1 -g 0.0
    ./scripts/ml/pretrain_EEGPT.py -a 1.0 -b 0.3 -g 0.0
    ./scripts/ml/pretrain_EEGPT.py -a 1.0 -b 1.0 -g 0.0
    ./scripts/ml/pretrain_EEGPT.py -a 1.0 -b 3.0 -g 0.0        
    """
