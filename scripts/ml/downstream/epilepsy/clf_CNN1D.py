#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-03-12 12:51:20 (ywatanabe)"

import sys

import matplotlib

matplotlib.use("Agg")
import random

import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from icecream import ic
from models.CNN1D import CNN1D
from models.EEGPT import load_trained_EEGPT
from models.LossBalancer import LossBalancer
from scripts.data.database.SQLiteDataset import load_ds_dls
from torch.optim import Adam


def _Mb2Tb(_mb, ilae_thres=2):
    ilaeb = _mb.reshape(-1)
    indi_lighter = ilaeb <= ilae_thres
    ilaeb[indi_lighter] = 0
    ilaeb[~indi_lighter] = 1
    tb = ilaeb.cuda()
    return tb


def base_step(
    model,
    model_sd,
    EEGPT,
    optimizer,
    optimizer_sd,
    lb,
    le,
    batch,
    i_epoch,
    i_batch,
    i_global,
    metrics,
    step="Training",
    alpha=1,
):

    color = {
        "Training": "cyan",
        "Validation": "green",
        "Test": "red",
    }[step]

    if CONFIG["IS_DEBUG"] and (i_batch > 50):
        return None

    if step == "Training":
        model.train()
    else:
        model.eval()

    Db, Sb, Xb, _Mb = batch

    # Target
    Tb = _Mb2Tb(_Mb)

    # X
    Xb = Xb.cuda()

    # Sb
    le.fit(Sb)
    Sb = torch.tensor(le.transform(Sb)).cuda()

    # Forwarding
    Lb = EEGPT(Xb, get_latent=True)
    logits = model(Lb)

    # Loss
    loss = F.cross_entropy(logits, Tb, reduction="none")
    try:
        loss = lb(loss, Tb, is_training=step == "Training").mean()
    except:
        return metrics

    if step == "Training":
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    pred_proba = F.softmax(logits, dim=-1)
    pred_class = pred_proba.argmax(dim=-1)
    bACC = mngs.ml.metrics.bACC(Tb, pred_class)

    loss = loss.detach().cpu().item()
    metrics["step"].append(step)
    metrics["i_global"].append(i_global)
    metrics["i_epoch"].append(i_epoch)
    metrics["i_batch"].append(i_batch)
    metrics["loss"].append(loss)
    metrics["bACC"].append(bACC)

    if i_batch == random.randint(0, 30) and (i_epoch % 10 == 0):
        mngs.gen.print_block(
            (
                f"{step}, Epoch# {i_epoch}, Batch #{i_batch}, Loss: {round(loss, 5)}, bACC: {round(bACC, 3)}"
                f"\nTb:\n{Tb}"
                f"\npred_class:\n{pred_class}"
            ),
            c=color,
        )
    return metrics


def train_and_eval(model, model_sd, dls, EEGPT, lb, le):

    optimizer = Adam(
        model.parameters(),
        lr=float(CONFIG["LEARNING_RATE"]),
        weight_decay=1e-5,
    )

    optimizer_sd = Adam(
        model_sd.parameters(),
        lr=float(CONFIG["LEARNING_RATE"]),
        weight_decay=1e-5,
    )

    metrics = mngs.gen.listed_dict()
    i_global = 0
    for i_epoch, epoch in enumerate(range(CONFIG["N_EPOCHS"])):

        for i_batch, batch in enumerate(dls["tra"]):
            base_step(
                model,
                model_sd,
                EEGPT,
                optimizer,
                optimizer_sd,
                lb,
                le,
                batch,
                i_epoch,
                i_batch,
                i_global,
                metrics,
                step="Training",
            )
            i_global += 1

        for i_batch, batch in enumerate(dls["val"]):
            base_step(
                model,
                model_sd,
                EEGPT,
                optimizer,
                optimizer_sd,
                lb,
                le,
                batch,
                i_epoch,
                i_batch,
                i_global,
                metrics,
                step="Validation",
            )

    # for i_batch, batch in enumerate(dls["tes"]):
    #     base_step(
    #         model,
    #         model_sd,
    #         EEGPT,
    #         optimizer,
    #         optimizer_sd,
    #         lb,
    #         le,
    #         batch,
    #         i_epoch,
    #         i_batch,
    #         i_global,
    #         metrics,
    #         step="Test",
    #     )

    return model, pd.DataFrame(metrics)


def subject_level_predict(ds, dls, step, EEGPT, model):
    model.eval()

    # Accumulates segment-level predictions
    pred_proba_all, T_all, S_all, M_all = [], [], [], []
    dl = dls[step]
    for batch in dl:
        Db, Sb, Xb, _Mb = batch

        Tb = _Mb2Tb(_Mb)
        Sb = np.array(Sb)
        Xb = Xb.cuda()

        # Prediction
        Lb = EEGPT(Xb, get_latent=True)
        logits = model(Lb)
        pred_proba = F.softmax(logits, dim=-1)

        # Buffering
        S_all.append(Sb)
        T_all.append(Tb.detach().cpu().numpy())
        pred_proba_all.append(pred_proba.detach().cpu().numpy())

    # To numpy
    pred_proba_all = np.vstack(pred_proba_all)
    S_all = np.hstack(S_all)
    T_all = np.hstack(T_all)

    color = {
        "tra": "cyan",
        "val": "green",
        "tes": "red",
    }[step]

    mngs.gen.print_block(mngs.gen.uq(T_all), c=color)

    # To subject-level predictions by taking means
    pred_proba_sub = []
    T_sub = []
    for s in np.unique(S_all):
        indi_s = S_all == s
        _pred_proba_sub = pred_proba_all[indi_s].mean(axis=0)
        pred_proba_sub.append(_pred_proba_sub)
        T_sub.append(T_all[indi_s][0])
    pred_proba_sub = np.vstack(pred_proba_sub)
    T_sub = np.hstack(T_sub)

    # Outputs
    pred_class_sub = pred_proba_sub.argmax(axis=-1)
    bACC = mngs.ml.metrics.bACC(T_sub, pred_class_sub)

    return (
        pred_proba_sub,
        pred_class_sub,
        T_sub,
        bACC,
    )


def main(eegpt_id):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds, dls = load_ds_dls("./config/DownStream/Epilepsy.yaml", from_cache=True)

    #   Unique Elements  Counts
    # 0               1  20,132
    # 1               3   3,593
    # 2               4   2,808
    # 3               5   4,577
    # 4               6   1,790

    # len(dls["tra"]) # 56
    # len(dls["val"]) # 37
    # len(dls["tes"]) # 37
    # len(dls["tra"].dataset) # 1800
    # len(dls["val"].dataset) # 1200
    # len(dls["tes"].dataset) # 1200

    ds_tes = dls["tes"].dataset
    for ii in range(len(ds_tes)):
        print(ds_tes[ii][-1])

    n_classes = 2
    model = CNN1D(n_classes, num_channels=64).to(device)
    n_classes_sd = 500
    model_sd = CNN1D(n_classes_sd, num_channels=64).to(device)

    EEGPT = load_trained_EEGPT(eegpt_id).to(device).eval()

    lb = LossBalancer(n_classes).to(device)
    le = mngs.ml.utils.LabelEncoder()

    model, metrics_df = train_and_eval(model, model_sd, dls, EEGPT, lb, le)

    # for step in ["tra", "val", "tes"]:
    for step in ["tra", "val"]:
        (pred_proba_sub, pred_class_sub, T_sub, bACC,) = subject_level_predict(
            ds,
            dls,
            step,
            EEGPT,
            model,
        )

        step_str = {
            "tra": "Training",
            "val": "Validation",
            "tes": "Test",
        }[step]

        labels = ["Lighter", "More severe"]

        fig, cm = mngs.ml.plt.conf_mat(
            plt,
            y_true=T_sub,
            y_pred=pred_class_sub,
            labels=labels,
            title=f"{step_str}\nbACC = {bACC:.3f}",
        )

        mngs.io.save(
            fig,
            CONFIG["SDIR"] + f"conf_mat/{step}_bACC_{bACC:.3f}.png",
        )

    fig = mngs.ml.plt.learning_curve(
        metrics_df, keys_to_plot=["loss", "bACC"]
    )  # , yscale="log"

    mngs.io.save(
        model,
        CONFIG["SDIR"] + f"checkpoint.pth",
    )
    mngs.io.save(metrics_df, CONFIG["SDIR"] + f"metrics.csv")
    for step in metrics_df.step.unique():
        mngs.io.save(
            metrics_df[metrics_df.step == step],
            CONFIG["SDIR"] + f"metrics_{step}.csv",
        )

    mngs.io.save(fig, CONFIG["SDIR"] + f"learning_curve.png")


if __name__ == "__main__":
    import argparse
    from glob import glob

    parser = argparse.ArgumentParser(
        description="Train a 1D CNN classifier on latent features extracted by a trained EEGPT model"
    )
    parser.add_argument(
        "--ID",
        default="FINISHED/2024Y-03M-01D-03h26m30s_EJgv",
        help="ID for the pretrained model",
    )
    args = parser.parse_args()
    print(f"\nargs:\n{args}")

    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        np=np,
        torch=torch,
    )
    plt, cc = mngs.plt.configure_mpl(plt, fig_scale=3)
    CONFIG["BATCH_SIZE"] = 32
    CONFIG["N_EPOCHS"] = 50
    # 100
    # /home/ywatanabe/proj/EEGPT/scripts/ml/downstream/epilepsy/clf_CNN1D/RUNNING/2024Y-03M-12D-12h40m45s_xFBd/

    main(eegpt_id=args.ID)

    mngs.gen.close(CONFIG)

    """
    ./scripts/ml/downstream/epilepsy/clf_CNN1D.py
    """
    # ----------------------------------------
    # Validation, Epoch# 50, Batch #14, Loss: 0.00663, bACC: 0.947
    # Tb:
    # tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #         0, 0, 0, 0, 0, 0, 0, 0], device='cuda:0')
    # pred_class:
    # tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    #         0, 1, 0, 0, 0, 0, 0, 0], device='cuda:0')
    # ----------------------------------------
