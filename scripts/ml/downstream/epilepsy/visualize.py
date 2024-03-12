#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-03-12 13:00:07 (ywatanabe)"


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
    _ILAE_all = []
    _L_all = []
    for batch in dls[step]:
        _, Sb, Xb, _Mb = batch
        ILAEb = np.array(ds.decode_meta(_Mb)["ILAE"]).astype(int)
        Lb = EEGPT(Xb.cuda(), get_latent=True)

        _S_all.append(Sb)
        _ILAE_all.append(ILAEb)
        _L_all.append(Lb.detach().cpu().numpy())

    _S_all = np.hstack(_S_all)
    _ILAE_all = np.hstack(_ILAE_all)
    _L_all = np.vstack(_L_all)

    # Cap the number of subjects
    S_all = []
    ILAE_all = []
    L_all = []
    for sub in np.unique(_S_all):

        indi_sub = np.where(_S_all == sub)[0]
        indi_sub = np.random.permutation(indi_sub)[:30]

        S_all.append(_S_all[indi_sub])
        ILAE_all.append(_ILAE_all[indi_sub])
        L_all.append(_L_all[indi_sub])

    S_all = np.hstack(S_all)
    #                                 Unique Elements Counts
    # 0   ds_bids_epilepsy_pros_adult_v1_sub-POUN0002     30
    # 1   ds_bids_epilepsy_pros_adult_v1_sub-POUN0014     30
    # 2   ds_bids_epilepsy_pros_adult_v1_sub-POUN0015     30
    # 3   ds_bids_epilepsy_pros_adult_v1_sub-POUN0022     30
    # 4   ds_bids_epilepsy_pros_adult_v1_sub-POUN0042     30
    # 5   ds_bids_epilepsy_pros_adult_v1_sub-POUN0047     30
    # 6   ds_bids_epilepsy_pros_adult_v1_sub-POUN0054     30
    # 7   ds_bids_epilepsy_pros_adult_v1_sub-POUN0056     30
    # 8   ds_bids_epilepsy_pros_adult_v1_sub-POUN0067     30
    # 9   ds_bids_epilepsy_pros_adult_v1_sub-POUN0074     30
    # 10  ds_bids_epilepsy_pros_adult_v1_sub-POUN0079     30
    # 11  ds_bids_epilepsy_pros_adult_v1_sub-POUN0088     30
    # 12  ds_bids_epilepsy_pros_adult_v1_sub-POUN0094     30
    # 13  ds_bids_epilepsy_pros_adult_v1_sub-POUN0119     30
    # 14   ds_bids_epilepsy_pros_adult_v1_sub-PSU0001     30
    # 15   ds_bids_epilepsy_pros_adult_v1_sub-PSU0002     30
    # 16   ds_bids_epilepsy_pros_adult_v1_sub-PSU0003     30
    # 17   ds_bids_epilepsy_pros_adult_v1_sub-PSU0004     30
    # 18   ds_bids_epilepsy_pros_adult_v1_sub-PSU0005     30
    # 19   ds_bids_epilepsy_pros_adult_v1_sub-PSU0007     30
    # 20   ds_bids_epilepsy_pros_adult_v1_sub-PSU0012     30
    # 21   ds_bids_epilepsy_pros_adult_v1_sub-PSU0015     30
    # 22   ds_bids_epilepsy_pros_adult_v1_sub-PSU0016     30
    # 23   ds_bids_epilepsy_pros_adult_v1_sub-PSU0020     30
    # 24   ds_bids_epilepsy_pros_adult_v1_sub-PSU0022     30
    # 25   ds_bids_epilepsy_pros_adult_v1_sub-PSU0028     30
    # 26   ds_bids_epilepsy_pros_adult_v1_sub-PSU0029     30
    # 27   ds_bids_epilepsy_pros_adult_v1_sub-PSU0030     30
    # 28   ds_bids_epilepsy_pros_adult_v1_sub-PSU0032     30
    # 29   ds_bids_epilepsy_pros_adult_v1_sub-PSU0037     30
    # 30   ds_bids_epilepsy_pros_adult_v1_sub-PSU0039     30
    # 31   ds_bids_epilepsy_pros_adult_v1_sub-PSU0047     30
    # 32   ds_bids_epilepsy_pros_adult_v1_sub-PSU0050     30

    ILAE_all = np.hstack(ILAE_all)
    #    Unique Elements Counts
    # 0                1    600
    # 1                3    150
    # 2                4    120
    # 3                5     60
    # 4                6     60

    def to_binary(ILAE_all, thres=2):
        indi_lighter = ILAE_all <= thres
        indi_more_severe = ~indi_lighter

        ILAE_all[indi_lighter] = 0
        ILAE_all[indi_more_severe] = 1

        return ILAE_all

    # ILAE_all = to_binary(ILAE_all)

    L_all = np.vstack(L_all)

    return {
        f"S_{step}": S_all,
        f"ILAE_{step}": ILAE_all,
        f"L_{step}": L_all,
    }


def to_L_all(ds, dls, EEGPT):
    out = {}
    # for step in ["tra", "val", "tes"]:
    for step in ["tra", "val"]:
        out.update(to_L(ds, dls, EEGPT, step))
    return out


def to_vec(x):
    return x.reshape(len(x), -1)
    # return x.mean(axis=-1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds, dls = load_ds_dls("./config/DownStream/Epilepsy.yaml", from_cache=True)

    EEGPT = load_trained_EEGPT(args.ID).to(device).eval()

    data = to_L_all(ds, dls, EEGPT)

    STEPS = ["tra", "val"]
    for label_str in ["S", "ILAE"]:
        label_type = {
            "S": "Subject",
            "ILAE": "ILAE",
        }[label_str]
        for supervised in [False, True]:
            supervised_str = "supervised" if supervised else "unsupervised"

            # ----------------------------------------
            # Embedding based on Training
            fig, legend_figs, _ = mngs.ml.clustering.umap(
                [to_vec(data[f"L_{step}"]) for step in STEPS],
                [data[f"{label_str}_{step}"] for step in STEPS],
                axes_titles=["Training", "Validation"],
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
                [np.vstack([to_vec(data[f"L_{step}"]) for step in STEPS])],
                [np.hstack([data[f"{label_str}_{step}"] for step in STEPS])],
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
    # /home/ywatanabe/proj/EEGPT/scripts/ml/downstream/epilepsy/clf_CNN1D/RUNNING/2024Y-03M-12D-12h40m45s_xFBd/

    main()

    mngs.gen.close(CONFIG)
    """
    ./scripts/ml/downstream/epilepsy/visualize.py
    """
