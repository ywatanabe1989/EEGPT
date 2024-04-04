#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-03-30 14:19:04 (ywatanabe)"


import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def to_L(ds, dls, EEGPT, step):

    # Converts to latent features
    _S_all = []
    _D_all = []
    _C_all = []
    _I_all = []
    _L_all = []
    for batch in dls[step]:
        _, Sb, Xb, _Mb = batch
        Db = ds.decode_meta(_Mb)["disease_type"]
        Cb = ds.decode_meta(_Mb)["cognitive_level"]
        Ib = ds.decode_meta(_Mb)["institution"]
        Lb = EEGPT(Xb.cuda(), get_latent=True)

        _S_all.append(Sb)
        _D_all.append(Db)
        _C_all.append(Cb)
        _I_all.append(Ib)
        _L_all.append(Lb.detach().cpu().numpy())

    _S_all = np.hstack(_S_all)
    _D_all = np.hstack(_D_all)
    _C_all = np.hstack(_C_all)
    _I_all = np.hstack(_I_all)
    _L_all = np.vstack(_L_all)

    # Cap the number of subjects
    S_all = []
    D_all = []
    C_all = []
    I_all = []
    L_all = []
    for sub in np.unique(_S_all):

        indi_sub = np.where(_S_all == sub)[0]
        indi_sub = np.random.permutation(indi_sub)[:100]

        S_all.append(_S_all[indi_sub])
        D_all.append(_D_all[indi_sub])
        C_all.append(_C_all[indi_sub])
        I_all.append(_I_all[indi_sub])
        L_all.append(_L_all[indi_sub])

    D_all = np.hstack(D_all)
    sort_order = {"HV": 0, "AD": 1, "DLB": 2, "NPH": 3}
    sorted_indices = np.argsort([sort_order[i] for i in D_all])
    D_all = D_all[sorted_indices]
    S_all = np.hstack(S_all)[sorted_indices]
    C_all = np.hstack(C_all)[sorted_indices]
    I_all = np.hstack(I_all)[sorted_indices]
    L_all = np.vstack(L_all)[sorted_indices]

    return {
        f"S_{step}": S_all,
        f"D_{step}": D_all,
        f"C_{step}": C_all,
        f"I_{step}": I_all,
        f"L_{step}": L_all,
    }


def to_L_all(ds, dls, EEGPT):
    out = {}
    for step in ["tra", "val", "tes"]:
        out.update(to_L(ds, dls, EEGPT, step))
    return out


def to_vec(x):
    return x.reshape(len(x), -1)
    # return x.mean(axis=-1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds, dls = load_ds_dls("./config/DownStream/Dementia.yaml", from_cache=True)

    EEGPT = load_trained_EEGPT(args.ID).to(device).eval()

    data = to_L_all(ds, dls, EEGPT)

    # to subject-mean
    data_all = mngs.gen.listed_dict()
    for step in ["tra", "val", "tes"]:
        S, D, C, I, L = [], [], [], [], []
        for sub in np.unique(data[f"S_{step}"]):
            indi = data[f"S_{step}"] == sub
            S.append(data[f"S_{step}"][indi][0])
            D.append(data[f"D_{step}"][indi][0])
            C.append(data[f"C_{step}"][indi][0])
            I.append(data[f"I_{step}"][indi][0])
            L.append(data[f"L_{step}"][indi].mean(axis=0, keepdims=True))
        data[f"S_{step}"] = np.hstack(S)
        data[f"D_{step}"] = np.hstack(D)
        data[f"C_{step}"] = np.hstack(C)
        data[f"I_{step}"] = np.hstack(I)
        data[f"L_{step}"] = np.vstack(L)

        data_all[f"S"].append(np.hstack(S))
        data_all[f"D"].append(np.hstack(D))
        data_all[f"C"].append(np.hstack(C))
        data_all[f"I"].append(np.hstack(I))
        data_all[f"L"].append(np.vstack(L))

    data["S_all"] = np.hstack(data_all["S"])
    data["D_all"] = np.hstack(data_all["D"])
    data["C_all"] = np.hstack(data_all["C"])
    data["I_all"] = np.hstack(data_all["I"])
    data["L_all"] = np.vstack(data_all["L"])

    for label_str in ["S", "D", "C", "I"]:
        label_type = {
            "S": "Subject",
            "D": "Disease Type",
            "C": "Cognitive Level",
            "I": "Institution",
        }[label_str]
        for supervised in [False, True]:
            supervised_str = "supervised" if supervised else "unsupervised"

            # ----------------------------------------
            # Embedding based on Training
            fig, legend_figs, _ = mngs.ml.clustering.umap(
                # [to_vec(data[f"L_all"])],
                # [to_vec(data[f"{label_str}_all"])],
                # axes_titles=["All"],
                [to_vec(data[f"L_{step}"]) for step in ["tra", "val", "tes"]],
                [
                    data[f"{label_str}_{step}"]
                    for step in ["tra", "val", "tes"]
                ],
                axes_titles=["Training", "Validation", "Test"],
                supervised=supervised,
                alpha=0.5,
                title=f"UMAP Embedding based on Training data (Labeled by {label_type})",
                s=10,
            )
            # Saves
            for i_legend, legend_fig in enumerate(legend_figs):
                mngs.io.save(
                    legend_fig,
                    CONFIG["SDIR"]
                    + f"UMAP/{supervised_str}/{label_type.replace(' ', '_').lower()}_legend/{i_legend}.png",
                )
            mngs.io.save(
                fig,
                CONFIG["SDIR"]
                + f"UMAP/{supervised_str}/separated/{label_type.replace(' ', '_').lower()}.png",
            )
            # ----------------------------------------

            # ----------------------------------------
            # Merged
            fig, legend_figs, _ = mngs.ml.clustering.umap(
                [
                    np.vstack(
                        [
                            to_vec(data[f"L_{step}"])
                            for step in ["tra", "val", "tes"]
                        ]
                    )
                ],
                [
                    np.hstack(
                        [
                            data[f"{label_str}_{step}"]
                            for step in ["tra", "val", "tes"]
                        ]
                    )
                ],
                axes_titles=["Merged"],
                supervised=supervised,
                alpha=0.5,
                title=f"UMAP Clustering (Labeled by {label_type})",
                s=10,
            )
            # Saves
            for i_legend, legend_fig in enumerate(legend_figs):
                mngs.io.save(
                    legend_fig,
                    CONFIG["SDIR"]
                    + f"UMAP/{supervised_str}/merged/{label_type.replace(' ', '_').lower()}_legend_{i_legend}.png",
                )
            mngs.io.save(
                fig,
                CONFIG["SDIR"]
                + f"UMAP/{supervised_str}/merged/{label_type.replace(' ', '_').lower()}.png",
            )
            # ----------------------------------------


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
    plt, CC = mngs.plt.configure_mpl(plt, fig_scale=3)

    # Overwrite CONFIG
    CONFIG["BATCH_SIZE"] = 32
    CONFIG["N_EPOCHS"] = 100

    main()

    mngs.gen.close(CONFIG)

    # ./scripts/ml/downstream/dementia/visualize.py
