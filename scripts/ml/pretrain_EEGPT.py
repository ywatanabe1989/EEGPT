#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-17 10:12:24 (ywatanabe)"

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
    commitment_loss, entropy_loss, prediction_loss = model(Xb)
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
    metrics["commitment_loss"].append(commitment_loss.item())
    metrics["entropy_loss"].append(entropy_loss.item())
    metrics["prediction_loss"].append(prediction_loss.item())

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
        print(f"\nmodel.VQ.assignment_probs:\n{model.VQ.assignment_probs[:5]}")

        print(
            f"\nmodel.VQ.assignment_probs.argmax:\n{mngs.gen.unique(model.VQ.assignment_probs.argmax(axis=-1).cpu().detach().numpy())}"
        )

    return metrics


def train_and_eval(model, dls):

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


def main(model_config):
    # Loads
    ds, dls = load_ds_dls("./config/Pretrain/OpenNeuro.yaml")
    len(ds.df["dataset_id"].unique())  # 78
    len(ds.df["subject_id"].unique())  # 6144

    # Main
    model = EEGPT(model_config).cuda()
    _ = mngs.ml.utils.check_params(model, show=True)
    model, metrics_df = train_and_eval(model, dls)
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
        "-cc",
        "--commit_coef",
        type=float,
        default=1.0,
        help="Weight for the commitment loss",
    )
    parser.add_argument(
        "-ec",
        "--entropy_coef",
        type=float,
        default=0.3,
        help="Weight for the entropy loss",
    )
    parser.add_argument(
        "-pc",
        "--pred_coef",
        type=float,
        default=0.0,
        help="Weight for the prediction loss",
    )
    parser.add_argument(
        "--softmax_type",
        default="softmax",
        choices=["softmax", "gumbel_softmax_soft", "gumbel_softmax_hard"],
        help="Type of softmax to be used in the VQ layer: 'softmax', 'gumbel_softmax_soft', or 'gumbel_softmax_hard'.",
    )
    # for learning transformer weights
    parser.add_argument(
        "--conv_vq_weights_path",
        default=None,
        help="Load convolution and vector quantization layers and freeze them.",
    )

    args = parser.parse_args()

    args.conv_vq_weights_path = "./scripts/ml/pretrain_EEGPT/2024-02-17-01-48_uDT2kC6q/checkpoint.pth"  # fixme

    print(f"\nargs:\n{args}")

    model_config = dict(
        VQ_num_embeddings=4,
        VQ_embedding_dim=16,
        VQ_softmax_type=args.softmax_type,
        VQ_weight_norm=False,
        VQ_initializer="uniform",
        loss_commit_coef=args.commit_coef,
        loss_entropy_coef=args.entropy_coef,
        loss_pred_coef=args.pred_coef,
        # load_conv_and_vq=args.load_conv_and_vq,
        conv_vq_weights_path=args.conv_vq_weights_path,
    )

    main(model_config)
