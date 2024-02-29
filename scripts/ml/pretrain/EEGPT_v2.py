#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-29 22:48:48 (ywatanabe)"

import os
import random
import sys

import matplotlib
import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
import torch
from models.EEGPT import EEGPT
from models.MultiTaskLoss import MultiTaskLoss
from scripts.data.database.SQLiteDataset import load_ds_dls
from torch.optim import SGD, Adam

matplotlib.use("Agg")


# Functions
def plot_prediction(model, title=None):
    true = model.post
    pred = model.y

    i_rand = random.randint(0, len(true) - 1)

    true = true[i_rand]
    pred = pred[i_rand]

    # Main
    fig, ax = mngs.plt.subplots()

    xx = np.arange(len(true))

    ax.plot(xx, true, color=CC["blue"], label="Ground Truth")
    ax.plot(xx, pred, color=CC["red"], label="Prediction")
    ax.legend()

    ax.set_xlabel("Feature index (among 14 channels and 6 frequency bands)")
    ax.set_ylabel("Band power ratio in a channel[a.u.]")
    ax.set_ylim(-1, 1)

    fig.suptitle(title)

    return fig


def remove_nan_inclusing_samples(x):
    # Remove samples with nan values
    indi_nan = x.isnan().any(-1).any(-1)
    has_nan = indi_nan.sum()
    x = x[~indi_nan]
    return x


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
    _Db, _Sb, Xb, _Mb = batch

    Xb = Xb.cuda()  # [-1, 14, 128*9]

    Xb = remove_nan_inclusing_samples(Xb)

    # Forward pass
    try:
        loss = model(Xb)
    except Exception as e:
        print("\nAn error occurred during the forward pass. Skipping.")
        print(e)
        return metrics

    if torch.isnan(loss):
        print("\nThe loss is NaN. Skipping.")
        return metrics

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
        fig = plot_prediction(
            model, title=f"An example prediction in epoch #{i_epoch:03d}"
        )
        mngs.gen.save(
            fig,
            CONFIG["SDIR"]
            + f"representative_predictions/{step}/epoch#{i_epoch:03d}.png",
        )
        print(
            f"\n{step}, Epoch# {i_epoch}, Batch #{i_batch}, Loss: {round(loss, 8)}"
        )

    return metrics


def train_and_eval(model, dls):

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


def main(model_config):
    # Loads
    ds, dls = load_ds_dls("./config/Pretrain/OpenNeuro.yaml", from_cache=True)
    print(f"N datasets: {len(ds.df['dataset_id'].unique())}")  # 78
    print(f"N subjects: {len(ds.df['subject_id'].unique())}")  # 6144

    # Main
    model = EEGPT(model_config).cuda()
    _ = mngs.ml.utils.check_params(model, show=True)
    model, metrics_df = train_and_eval(model, dls)
    fig = mngs.ml.plt.learning_curve(metrics_df, yscale="log")

    # Saves
    prefix = f"{model_config['pretrain_stage']}"
    mngs.io.save(
        model,
        CONFIG["SDIR"] + f"{prefix}_checkpoint.pth",
    )
    mngs.io.save(metrics_df, CONFIG["SDIR"] + f"{prefix}_metrics.csv")
    mngs.io.save(fig, CONFIG["SDIR"] + f"{prefix}_learning_curve.png")
    mngs.gen.save(model_config, CONFIG["SDIR"] + f"{prefix}_model_config.yaml")


if __name__ == "__main__":
    from scripts.ml.parse_model_configs import parse_model_configs

    # Configurations
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        os=os,
        random=random,
        torch=torch,
        np=np,
    )
    WINDOW_SIZE_PTS = CONFIG["WINDOW_SIZE_SEC"] * CONFIG["TGT_FS"]
    # CONFIG["BATCH_SIZE"] = 32

    model_config = parse_model_configs()

    # # 1st pretraining for the Convolution and Quantization layers
    # model_config = parse_model_configs(pretrain_stage="conv_and_vq")
    main(model_config)

    # # 2nd Pretraining for the Transformer layers
    # model_config = parse_model_configs(pretrain_stage="transformer")
    # model_config["conv_vq_weights_path"] = (
    #     CONFIG["SDIR"] + "conv_and_vq_checkpoint.pth"
    # )
    # main(model_config)

    # # mngs.gen.save(model_config, CONFIG["SDIR"] + "model_config.yaml")
    # # Closes
    # mngs.gen.close(CONFIG)
