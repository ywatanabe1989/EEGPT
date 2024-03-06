#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "

import random

from models.EEGConvAE import EEGConvAE
from scripts.data.database.SQLiteDataset import load_ds_dls


def plot_reconstructed_sig(dl_tes, model, title="Title here"):
    batch = next(iter(dl_tes))
    Xb = batch[2][..., 64:-64]
    indi_nan = np.isnan(Xb).any(dim=-1).any(dim=-1).bool()
    Xb = Xb[~indi_nan]

    # Forward pass
    y = model(Xb)

    # To z
    Xb = model.to_z(Xb)

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
    fig, axes = mngs.plt.subplots(nrows=3)  # , sharex=True, sharey=True
    for i_ch in range(len(_xb)):
        axes[0].plot(xx, _xb[i_ch], label=f"ch#{i_ch}")
        axes[1].plot(xx, _y[i_ch], label=f"ch#{i_ch}")
        axes[2].plot(xx, _diff[i_ch], label=f"ch#{i_ch}")
    fig.suptitle(title)
    axes[0].set_ylabel("Original")
    axes[1].set_ylabel("Reconstructed")
    axes[2].set_ylabel(f"Diff.\n(Original - Reconstructed)")
    return fig


if __name__ == "__main__":
    from scripts.ml.parse_model_configs import parse_model_configs

    # Configurations
    CONFIG, sys.stdout, sys.stderr, plt, cc = mngs.gen.start(
        sys,
        plt,
        # os=os,
        random=random,
        # torch=torch,
        # np=np,
    )
    WINDOW_SIZE_PTS = CONFIG["WINDOW_SIZE_SEC"] * CONFIG["TGT_FS"]

    # Loads Pretrained model
    model_config = parse_model_configs(pretrain_stage="AE")
    # ID = "2024Y-02M-27D-19h29m53s_JQpg"
    ID = "2024Y-02M-27D-22h23m09s_54wb"
    lpath = f"./scripts/ml/pretrain_EEGConvAE/FINISHED/{ID}/AE_checkpoint.pth"

    model = EEGConvAE()
    model.load_state_dict(mngs.io.load(lpath).state_dict())
    model.eval()

    # Loads test data
    ds, dls = load_ds_dls("./config/Pretrain/OpenNeuro.yaml", from_cache=True)

    import matplotlib

    matplotlib.use("Agg")
    fig = plot_reconstructed_sig(dls["tes"], model)
    mngs.io.save(fig, "/tmp/test.png")
