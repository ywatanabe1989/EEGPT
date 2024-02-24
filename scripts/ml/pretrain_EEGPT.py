#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-23 01:54:23 (ywatanabe)"

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
from scripts.data.database.SQLiteDataset import load_ds_dls
from torch.optim import SGD

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

    # Normalization after cropping
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

    # General
    parser.add_argument(
        "--learning_rate",
        default=1e3,
        type=float,
    )

    parser.add_argument(
        "--optimizer",
        default="SGD",
    )

    # Conv and VQ
    parser.add_argument(
        "--conv_vq_weights_path",
        default=None,
    )

    # VQ parameters
    parser.add_argument("--vq_num_embeddings", type=int, default=4)
    parser.add_argument("--vq_embedding_dim", type=int, default=16)
    parser.add_argument(
        "--vq_softmax_type", type=str, default="gumbel_softmax_hard"
    )
    parser.add_argument("--vq_commit_coef", type=float, default=1.0)
    parser.add_argument("--vq_entropy_coef", type=float, default=0.3)
    parser.add_argument("--vq_pred_coef", type=float, default=0.0)

    # Transformer parameters
    parser.add_argument("--tf_n_heads", type=int, default=2)
    parser.add_argument("--tf_hidden_dim", type=int, default=16)
    parser.add_argument("--tf_n_enc_layers", type=int, default=2)
    parser.add_argument("--tf_n_dec_layers", type=int, default=2)
    parser.add_argument("--tf_dropout_ratio", type=float, default=0.1)

    # parser.add_argument(
    #     "-cc",
    #     "--commit_coef",
    #     type=float,
    #     default=1.0,
    #     help="Weight for the commitment loss",
    # )
    # parser.add_argument(
    #     "-ec",
    #     "--entropy_coef",
    #     type=float,
    #     default=0.3,
    #     help="Weight for the entropy loss",
    # )
    # parser.add_argument(
    #     "-pc",
    #     "--pred_coef",
    #     type=float,
    #     default=0.0,
    #     help="Weight for the prediction loss",
    # )
    # parser.add_argument(
    #     "--num_embeddings",
    #     default=4,
    #     type=int,
    #     help="The number of embeddings to be uesd for quantization",
    # )

    # parser.add_argument(
    #     "--softmax_type",
    #     default="softmax",
    #     choices=["softmax", "gumbel_softmax_soft", "gumbel_softmax_hard"],
    #     help="Type of softmax to be used in the VQ layer: 'softmax', 'gumbel_softmax_soft', or 'gumbel_softmax_hard'.",
    # )

    # parser.add_argument("--vq_commit_coef", type=float, default=1.0)
    # parser.add_argument("--vq_entropy_coef", type=float, default=0.3)
    # parser.add_argument("--vq_pred_coef", type=float, default=0.0)

    # parser.add_argument("--tf_n_heads", type=int, default=8)
    # parser.add_argument("--tf_hidden_dim", type=int, default=512)
    # parser.add_argument("--tf_n_enc_layers", type=int, default=6)
    # parser.add_argument("--tf_n_dec_layers", type=int, default=6)
    # parser.add_argument("--tf_dropout_ratio", type=float, default=0.1)

    args = parser.parse_args()

    print(f"\nargs:\n{args}")

    model_config = dict(
        VQ_num_embeddings=args.num_embeddings,
        VQ_embedding_dim=16,
        VQ_softmax_type=args.softmax_type,
        VQ_weight_norm=False,
        VQ_initializer="uniform",
        loss_commit_coef=args.commit_coef,
        loss_entropy_coef=args.entropy_coef,
        loss_pred_coef=args.pred_coef,
        conv_vq_weights_path=args.conv_vq_weights_path,
    )

    if args.conv_vq_weights_path:
        import ipdb

        ipdb.set_trace()

    main(model_config)

    mngs.gen.save(model_config, CONFIG["SDIR"] + "model_config.yaml")
    mngs.gen.close(CONFIG)

    # 2024-02-17-19-55_0qjaeWlU_04
    # 2024-02-17-19-55_eMqdWxV3_16
    # 2024-02-17-19-55_y49afhEJ_08
