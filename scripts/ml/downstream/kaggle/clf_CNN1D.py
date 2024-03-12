#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-03-11 14:14:19 (ywatanabe)"

import sys

import matplotlib

matplotlib.use("Agg")
import warnings

import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from models.CNN1D import CNN1D
from models.EEGPT import load_trained_EEGPT
from models.LossBalancer import LossBalancer
from scripts.ml.downstream.kaggle.SQLiteDataset import load_ds_dls
from torch.optim import Adam

CONFIG = mngs.gen.load_configs()


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
    if CONFIG["IS_DEBUG"] and (i_batch > 50):
        return None

    if step == "Training":
        model.train()
    else:
        model.eval()

    Sb, Eb, Xb, Vb = batch

    Xb = Xb.cuda()
    Vb = Vb.cuda()

    le.fit(Sb)

    Sb = torch.tensor(le.transform(Sb)).cuda()

    Lb = EEGPT(Xb, get_latent=True)

    logits = model(Lb)
    input = F.log_softmax(logits, dim=-1)
    pred_proba = F.softmax(logits, dim=-1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        loss = F.kl_div(input, Vb, reduction="batchmean")

    if step == "Training":
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss = loss.detach().cpu().item()
    metrics["step"].append(step)
    metrics["i_global"].append(i_global)
    metrics["i_epoch"].append(i_epoch)
    metrics["i_batch"].append(i_batch)
    metrics["loss"].append(loss)

    if i_batch == 0:
        print(f"\n{'-'*40}\n")
        print(
            f"\n{step}, Epoch# {i_epoch}, Batch #{i_batch}, Loss: {round(loss, 5)}"
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
        # for i_epoch, epoch in enumerate(range(10)):

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

    for i_batch, batch in enumerate(dls["tes"]):
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
            step="Test",
        )

    return model, pd.DataFrame(metrics)


def subject_level_predict(ds, dls, step, EEGPT, model):
    model.eval()

    # Accumulates segment-level predictions
    pred_proba_all, T_all, S_all, E_all = [], [], [], []
    for batch in dls[step]:

        Sb, Eb, Xb, Vb = batch

        Xb = Xb.cuda()
        Vb = Vb.cuda()

        # Prediction
        Lb = EEGPT(Xb, get_latent=True)
        logits = model(Lb)
        pred_proba = F.softmax(logits, dim=-1)

        # Buffering
        S_all.append(Sb)  # Subject id
        E_all.append(Eb)  # EEG id
        pred_proba_all.append(pred_proba.detach().cpu().numpy())
        T_all.append(Vb.detach().cpu().numpy())

    # To numpy
    pred_proba_all = np.vstack(pred_proba_all)
    S_all = np.hstack(S_all)
    T_all = np.vstack(T_all)
    E_all = np.hstack(E_all)

    pred_proba_eeg = []
    E_eeg = []
    for eeg_id in np.unique(E_all):
        indi_e = E_all == eeg_id
        _pred_proba_eeg = pred_proba_all[indi_e].mean(axis=0)
        pred_proba_eeg.append(_pred_proba_eeg)
        E_e = E_all[indi_e]
        assert len(np.unique(E_e)) == 1
        E_eeg.append(E_e[0])

    pred_proba_eeg = np.vstack(pred_proba_eeg)
    E_eeg

    submission = []
    for i_eeg in range(len(E_eeg)):
        cols_vote = [
            "seizure_vote",
            "lpd_vote",
            "gpd_vote",
            "lrda_vote",
            "grda_vote",
            "other_vote",
        ]
        _submission = {
            "eeg_id": E_eeg[i_eeg],
            **{
                col: vote
                for col, vote in zip(cols_vote, pred_proba_eeg[i_eeg])
            },
        }
        submission.append(pd.DataFrame(pd.Series(_submission)))

    submission = pd.concat(submission).T

    return submission


def main(eegpt_id):
    import sys

    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        np=np,
        torch=torch,
    )
    plt, cc = mngs.plt.configure_mpl(plt, fig_scale=3)
    CONFIG["BATCH_SIZE"] = 32
    CONFIG["N_EPOCHS"] = 100

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds, dls = load_ds_dls("./config/DownStream/Kaggle.yaml", from_cache=True)
    ds_tes, dls_tes = load_ds_dls(
        "./config/DownStream/Kaggle_Test.yaml",
        from_cache=True,
        batch_size=2,
    )
    len(dls_tes["tes"])

    # n_classes, i_T = get_label_info(batch=next(iter(dls["tra"])), ds=ds)
    n_classes = 6
    model = CNN1D(n_classes, num_channels=64).to(device)
    n_classes_sd = 500
    model_sd = CNN1D(n_classes_sd, num_channels=64).to(device)

    EEGPT = load_trained_EEGPT(eegpt_id).to(device).eval()

    lb = LossBalancer(n_classes).to(device)
    le = mngs.ml.utils.LabelEncoder()

    model, metrics_df = train_and_eval(model, model_sd, dls, EEGPT, lb, le)

    # Predict the test data for submission
    step = "tes"
    submission = subject_level_predict(
        ds_tes,
        dls_tes,
        step,
        EEGPT,
        model,
    )

    # Saves outputs
    # submission
    mngs.io.save(submission, CONFIG["SDIR"] + "submission.csv")
    # Learning Curves
    fig_lc = mngs.ml.plt.learning_curve(
        metrics_df, keys_to_plot=["loss"], yscale="log"
    )
    mngs.io.save(fig_lc, CONFIG["SDIR"] + f"learning_curve.png")

    # Model weights
    mngs.io.save(
        model,
        CONFIG["SDIR"] + f"checkpoint.pth",
    )
    # Metrics
    mngs.io.save(metrics_df, CONFIG["SDIR"] + f"metrics/all.csv")
    for step in metrics_df.step.unique():
        mngs.io.save(
            metrics_df[metrics_df.step == step],
            CONFIG["SDIR"] + f"metrics/{step}.csv",
        )

    # Close
    mngs.gen.close(CONFIG)


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

    main(eegpt_id=args.ID)

    """
    sb kaggle ./scripts/ml/downstream/kaggle/clf_CNN1D.py
    """
