#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-03-12 13:30:55 (ywatanabe)"

import sys

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from models.CNN1D import CNN1D
from models.EEGPT import load_trained_EEGPT
from models.LossBalancer import LossBalancer
from scripts.data.database.SQLiteDataset import load_ds_dls
from torch.optim import Adam


def sort_disease_types(diags_list):
    diags_list = diags_list.copy()
    order = ["HV", "AD", "DLB", "NPH"]

    _reordered_order = []
    for d in diags_list:
        try:
            i, m = mngs.general.grep(order, d)
            assert len(i) == 1
            _reordered_order.append(i[0])
        except Exception as e:
            print(e)

    reordered_order = np.argsort(_reordered_order)
    return reordered_order  # [diags_list[i] for i in reordered_order]


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def get_label_info(batch, ds, T_str="disease_type"):
    _Db, Sb, Xb, Mb = batch
    i_T = np.where(ds.decode_meta(Mb).columns == T_str)[0][0]
    Tb = Mb[:, i_T]
    n_classes = len(np.unique(Tb))
    return n_classes, i_T


def base_step(
    model,
    model_sd,
    EEGPT,
    optimizer,
    optimizer_sd,
    lb,
    le,
    batch,
    i_T,
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

    _, Sb, Xb, _Mb = batch

    # indi_nan = Xb.isnan().any(-1).any(-1)

    Tb = _Mb[:, i_T].cuda()  # [~indi_nan]
    Xb = Xb.float().cuda()  # [~indi_nan]

    le.fit(Sb)
    # print(f"LE classes: {len(le.classes_)}")
    Sb = torch.tensor(le.transform(Sb)).cuda()  # [~indi_nan]

    Lb = EEGPT(Xb, get_latent=True)

    logits = model(Lb)
    loss = F.cross_entropy(logits, Tb)

    # logits_sd = model_sd(Lb.detach())
    # loss_sd = F.cross_entropy(logits_sd, Sb)

    # # Trains the subject discriminator
    # if step == "Training":
    #     optimizer_sd.zero_grad()
    #     loss_sd.backward()
    #     optimizer_sd.step()

    # # Calculates the loss for the subject discrimination task
    # # with torch.no_grad():
    # logits_sd = model_sd(Lb)
    # loss_adv = -F.cross_entropy(logits_sd, Sb)

    # # logits_sd = model_sd(Lb)
    # loss_sd = F.cross_entropy(logits_sd, Sb)
    # contrastive_loss = loss / (alpha * loss_sd)

    # total_loss = loss + alpha * loss_adv

    if step == "Training":
        optimizer.zero_grad()
        loss.backward()
        # contrastive_loss.backward()  # [REVISED]
        # total_loss.backward()  # [REVISED]
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

    if i_batch == 0:
        print(f"\n{'-'*40}\n")
        print(
            f"\n{step}, Epoch# {i_epoch}, Batch #{i_batch}, Loss: {round(loss, 5)}, bACC: {round(bACC, 3)}"
        )
        # print(f"Loss_sd: {round(loss_sd.detach().cpu().numpy().item(), 5)}")
        # print(
        #     f"contrastive_sd: {round(contrastive_loss.detach().cpu().numpy().item(), 5)}"
        # )
        # print("\nTb:")
        # print(mngs.gen.unique(Tb.detach().cpu().numpy()))
        # print("\npred_class:")
        # print(mngs.gen.unique(pred_class.detach().cpu().numpy()))

    return metrics


def train_and_eval(model, model_sd, dls, i_T, EEGPT, lb, le):

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
                i_T,
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
                i_T,
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
            i_T,
            i_epoch,
            i_batch,
            i_global,
            metrics,
            step="Test",
        )

    return model, pd.DataFrame(metrics)


def subject_level_predict(ds, dls, step, EEGPT, model, i_T, without_HV=False):
    model.eval()

    # Accumulates segment-level predictions
    pred_proba_all, T_all, S_all, M_all = [], [], [], []
    for batch in dls[step]:
        _, Sb, Xb, _Mb = batch

        Sb = np.array(Sb)
        Xb = Xb.cuda()
        Tb = _Mb[:, i_T].cuda()

        # Prediction
        Lb = EEGPT(Xb, get_latent=True)
        logits = model(Lb)
        pred_proba = F.softmax(logits, dim=-1)

        # Buffering
        M_all.append(_Mb)
        S_all.append(Sb)
        T_all.append(Tb.detach().cpu().numpy())
        pred_proba_all.append(pred_proba.detach().cpu().numpy())

    # To numpy
    pred_proba_all = np.vstack(pred_proba_all)
    S_all = np.hstack(S_all)
    T_all = np.hstack(T_all)
    M_all = np.vstack(M_all)

    # T to labels
    labels = (
        pd.DataFrame(T_all, ds.decode_meta(M_all)["disease_type"])
        .drop_duplicates()[0]
        .to_dict()
    )

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

    # Drops HV-related data for three-class classification tasks
    if without_HV:
        excessive_prediction_classes = set(
            np.arange(pred_proba_sub.shape[1])
        ) - set(labels.values())

        if len(excessive_prediction_classes) == 1:
            ii_ex = list(excessive_prediction_classes)[0]

        if len(excessive_prediction_classes) == 0:
            ii_ex = labels.get("HV")

        # Mask the prediction for HV and update posterior probability
        cols_to_keep = np.arange(pred_proba_sub.shape[1]) != ii_ex
        pred_proba_sub = pred_proba_sub[:, cols_to_keep]
        pred_proba_sub /= pred_proba_sub.sum(axis=-1, keepdims=True)

        # Drops HV samples
        indi_HV = T_sub == ii_ex
        pred_proba_sub = pred_proba_sub[~indi_HV]
        T_sub = T_sub[~indi_HV]

        labels.pop("HV", None)

        assert len(pred_proba_sub) == len(T_sub)
        assert pred_proba_sub.shape[-1] == len(labels)

    # Outputs
    pred_class_sub = pred_proba_sub.argmax(axis=-1)
    bACC = mngs.ml.metrics.bACC(T_sub, pred_class_sub)

    return (
        pred_proba_sub,
        pred_class_sub,
        T_sub,
        labels,
        bACC,
    )


def main(eegpt_id):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds, dls = load_ds_dls("./config/DownStream/Dementia.yaml", from_cache=True)

    n_classes, i_T = get_label_info(batch=next(iter(dls["tra"])), ds=ds)
    model = CNN1D(n_classes, num_channels=64).to(device)
    n_classes_sd = 500
    model_sd = CNN1D(n_classes_sd, num_channels=64).to(device)

    EEGPT = load_trained_EEGPT(eegpt_id).to(device).eval()

    lb = LossBalancer(n_classes).to(device)
    le = mngs.ml.utils.LabelEncoder()

    model, metrics_df = train_and_eval(
        model, model_sd, dls, i_T, EEGPT, lb, le
    )

    for step in ["val", "tes"]:
        for without_HV in [False, True]:
            (
                pred_proba_sub,
                pred_class_sub,
                T_sub,
                labels,
                bACC,
            ) = subject_level_predict(
                ds,
                dls,
                step,
                EEGPT,
                model,
                i_T,
                without_HV=without_HV,
            )

            labels = list(pd.Series(labels).sort_values().index)

            indi_sorted = sort_disease_types(labels)
            sorted_labels = list(np.array(labels)[indi_sorted])

            step_str = {
                "val": "Validation (Osaka Univ.)",
                "tes": "Test (Kochi Univ & Nippon Life Hospital)",
            }[step]

            fig, cm = mngs.ml.plt.conf_mat(
                plt,
                y_true=T_sub,
                y_pred=pred_class_sub,
                labels=labels,
                sorted_labels=sorted_labels,
                title=f"{step_str}\nbACC = {bACC:.3f}",
            )

            mngs.io.save(
                fig,
                CONFIG["SDIR"]
                + f"conf_mat/{step}_{'wo_HV' if without_HV else 'w_HV'}.png",
            )

    fig = mngs.ml.plt.learning_curve(metrics_df, yscale="log")

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
    CONFIG["N_EPOCHS"] = 100

    main(eegpt_id=args.ID)

    mngs.gen.close(CONFIG)

    # # OUTPUTS
    # 4 classes
    # 0.624
    # 0.463
    # 3 classes
    # 0.589
    # 0.452
