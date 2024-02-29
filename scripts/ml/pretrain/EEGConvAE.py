#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-28 12:23:19 (ywatanabe)"

import os
import random
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.EEGPT import EEGPT

# from models.EEGConvAE import EEGConvAE
from models.MultiTaskLoss import MultiTaskLoss
from scripts.data.database.SQLiteDataset import load_ds_dls
from torch.optim import SGD, Adam

# Functions


def plot_reconstructed_sig(dl_tes, model, colors, title="Title here"):
    model.eval()

    batch = next(iter(dl_tes))
    Xb = batch[2][..., 64:-64]
    indi_nan = np.isnan(Xb).any(dim=-1).any(dim=-1).bool()
    Xb = Xb[~indi_nan].cuda()

    # Forward pass
    y = model(Xb)

    # To z
    Xb = model.to_z(Xb).abs()

    # To numpy
    Xb = Xb.detach().cpu().numpy()
    y = y.detach().cpu().numpy()

    # Picks a sample to plot
    i_sample = random.randint(0, len(Xb))
    _xb = Xb[i_sample]
    _y = y[i_sample]
    _diff = _xb - _y

    # x axis
    xx = np.linspace(0, 1, CONFIG["TGT_FS"])

    # Plots
    fig, axes = mngs.plt.subplots(nrows=3, sharex=True, sharey=True)
    for i_ch in range(len(_xb)):
        color_str = list(colors.keys())[i_ch % len(colors)]
        axes[0].plot(
            xx, _xb[i_ch], label=f"ch#{i_ch}", color=colors[color_str]
        )
        axes[1].plot(xx, _y[i_ch], label=f"ch#{i_ch}", color=colors[color_str])
        axes[2].plot(
            xx, _diff[i_ch], label=f"ch#{i_ch}", color=colors[color_str]
        )
    fig.suptitle(title)
    axes[0].set_ylabel("Original")
    axes[1].set_ylabel("Reconstructed")
    axes[2].set_ylabel(f"Diff.\n(Original - Reconstructed)")
    return fig


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

    # Normalization after cropping
    Xb = Xb.float().cuda()

    # Remove samples with nan values
    indi_nan = Xb.isnan().any(-1).any(-1)
    has_nan = indi_nan.sum()

    Xb = Xb[~indi_nan]

    # Forward pass
    start = random.randint(0, Xb.shape[-1] - 128 - 1)
    end = start + 128
    Xb = Xb[..., start:end]

    y = model(Xb)

    # loss = F.mse_loss(Xb, y)
    loss = ((Xb - y).abs() + 1e-5).log().mean()
    # loss = torch.log(torch.abs(x - y) + 1e-5).mean()

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
    # metrics["prediction_loss"].append(model.prediction_loss.item())
    # metrics["commitment_loss"].append(model.commitment_loss.item())
    # metrics["entropy_loss"].append(model.entropy_loss.item())
    # metrics["diversity_loss"].append(model.diversity_loss.item())

    # Print
    if i_batch == 0:
        print(f"\n{'-'*40}\n")
        print(
            f"\n{step}, Epoch# {i_epoch}, Batch #{i_batch}, Loss: {round(loss, 8)}"
        )

        # print(f"\nprediction loss: {model.prediction_loss.item():.5f}")
        # print(f"\ncommitment loss: {model.commitment_loss.item():.5f}")
        # print(f"\nentropy loss: {model.entropy_loss.item():.5f}")
        # print(f"\ndiversity loss: {model.diversity_loss.item():.5f}")

        # print(
        #     f"\nmodel.vq.embedding.weight.data:\n{model.vq.embedding.weight.data[:5]}"
        # )
        # print(f"\nmodel.vq.assignment_probs:\n{model.vq.assignment_probs[:5]}")

        # print(
        #     f"\nmodel.vq.assignment_probs.argmax:\n{mngs.gen.unique(model.vq.assignment_probs.argmax(axis=-1).cpu().detach().numpy())}"
        # )

    return metrics


def train_and_eval(model, dls, colors):

    # optimizer = SGD(model.parameters(), lr=float(CONFIG["LEARNING_RATE"]))
    optimizer = Adam(model.parameters(), lr=float(CONFIG["LEARNING_RATE"]))

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
        fig = plot_reconstructed_sig(
            dls["tes"], model, colors, title=f"epoch #{i_epoch}"
        )
        mngs.io.save(fig, CONFIG["SDIR"] + f"reconstructed/{i_epoch:03d}.png")
        plt.close()

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


def main(model_config, colors):
    # Loads
    ds, dls = load_ds_dls("./config/Pretrain/OpenNeuro.yaml", from_cache=True)
    len(ds.df["dataset_id"].unique())  # 78
    len(ds.df["subject_id"].unique())  # 6144

    # Main
    model = EEGPT(model_config).cuda()
    import ipdb

    ipdb.set_trace()
    batch = next(iter(dls["tra"]))
    # model = EEGConvAE().cuda()
    _ = mngs.ml.utils.check_params(model, show=True)
    model, metrics_df = train_and_eval(model, dls, colors)
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
    CONFIG, sys.stdout, sys.stderr, plt, cc = mngs.gen.start(
        sys,
        plt,
        os=os,
        random=random,
        torch=torch,
        np=np,
    )
    WINDOW_SIZE_PTS = CONFIG["WINDOW_SIZE_SEC"] * CONFIG["TGT_FS"]

    # # 1st pretraining for the Convolution and Quantization layers
    model_config = parse_model_configs()
    main(model_config, cc)

    # # 2nd Pretraining for the Transformer layers
    # model_config = parse_model_configs(pretrain_stage="transformer")
    # model_config["conv_vq_weights_path"] = (
    #     CONFIG["SDIR"] + "conv_and_vq_checkpoint.pth"
    # )
    # main(model_config)

    # mngs.gen.save(model_config, CONFIG["SDIR"] + "model_config.yaml")
    # Closes
    mngs.gen.close(CONFIG)

    """
    ./scripts/ml/pretrain_EEGConvAE.py
    """
