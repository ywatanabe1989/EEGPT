#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-03-01 12:14:35 (ywatanabe)"

import numpy as np
import torch
from scripts.data.database.SQLiteDataset import load_ds_dls  # SQLiteDataset

# from scripts.ml.downstream.Dementia.utils import get_latent, to_sub_mean
from tqdm import tqdm


def load_transfered_data(EEGPT_model, n_epochs=10, batch_size=64):
    ds, dls = load_ds_dls(
        "./config/DownStream/Dementia.yaml",
        from_cache=True,
        batch_size=batch_size,
    )

    # Converts to latent features
    X_tra, L_tra, D_tra, C_tra, S_tra = get_latent(
        ds,
        dls["tra"],
        EEGPT_model,
        n_epochs=n_epochs,
    )
    X_val, L_val, D_val, C_val, S_val = get_latent(
        ds,
        dls["val"],
        EEGPT_model,
        n_epochs=n_epochs,
    )
    X_tes, L_tes, D_tes, C_tes, S_tes = get_latent(
        ds,
        dls["tes"],
        EEGPT_model,
        n_epochs=n_epochs,
    )
    Xs_tra, Ls_tra, Ds_tra, Cs_tra, Ss_tra = to_sub_mean(
        X_tra, L_tra, D_tra, C_tra, S_tra
    )
    Xs_val, Ls_val, Ds_val, Cs_val, Ss_val = to_sub_mean(
        X_val, L_val, D_val, C_val, S_val
    )
    Xs_tes, Ls_tes, Ds_tes, Cs_tes, Ss_tes = to_sub_mean(
        X_tes, L_tes, D_tes, C_tes, S_tes
    )

    return dict(
        # Training
        X_tra=X_tra,
        L_tra=L_tra,
        D_tra=D_tra,
        C_tra=C_tra,
        S_tra=S_tra,
        Xs_tra=Xs_tra,
        Ls_tra=Ls_tra,
        Ds_tra=Ds_tra,
        Cs_tra=Cs_tra,
        Ss_tra=Ss_tra,
        # Validation
        X_val=X_val,
        L_val=L_val,
        D_val=D_val,
        C_val=C_val,
        S_val=S_val,
        Xs_val=Xs_val,
        Ls_val=Ls_val,
        Ds_val=Ds_val,
        Cs_val=Cs_val,
        Ss_val=Ss_val,
        # Test
        X_tes=X_tes,
        L_tes=L_tes,
        D_tes=D_tes,
        C_tes=C_tes,
        S_tes=S_tes,
        Xs_tes=Xs_tes,
        Ls_tes=Ls_tes,
        Ds_tes=Ds_tes,
        Cs_tes=Cs_tes,
        Ss_tes=Ss_tes,
    )


def get_latent(ds, dl, EEGPT_model, n_epochs=10):
    def _get_latent(dl, EEGPT_model):
        X_all = []
        L_all = []
        D_all = []
        C_all = []
        S_all = []

        EEGPT_model.eval()
        for batch in dl:
            Db, Sb, Xb, _Mb = batch
            meta = ds.decode_meta(_Mb)

            Lb = EEGPT_model(Xb.cuda(), get_latent=True).cpu().detach().numpy()

            X_all.append(Xb.cpu().detach().numpy())
            L_all.append(Lb)
            S_all.append(Sb)
            D_all.append(meta["disease_type"])
            C_all.append(meta["cognitive_level"])

        X_all = np.vstack(X_all)
        L_all = np.vstack(L_all)
        D_all = np.hstack(D_all)
        C_all = np.hstack(C_all)
        S_all = np.hstack(S_all)

        assert (
            len(X_all) == len(L_all) == len(D_all) == len(C_all) == len(S_all)
        )

        return (
            X_all,
            L_all,
            D_all,
            C_all,
            S_all,
        )

    # Initialize lists to store results for each epoch
    X_all, L_all, D_all, C_all, S_all = (
        [],
        [],
        [],
        [],
        [],
    )

    # Loop over 100 epochs to collect results
    for epoch in tqdm(range(n_epochs)):
        (x, l, d, c, s) = _get_latent(dl, EEGPT_model)

        # Append the results of this epoch to the lists
        X_all.append(x)
        L_all.append(l)
        D_all.append(d)
        C_all.append(c)
        S_all.append(s)

    X_all = np.vstack(X_all)
    L_all = np.vstack(L_all)
    D_all = np.hstack(D_all)
    C_all = np.hstack(C_all)
    S_all = np.hstack(S_all)

    return X_all, L_all, D_all, C_all, S_all


def to_sub_mean(_, L, D, C, S):
    S_sub = []
    L_sub = []
    D_sub = []
    C_sub = []
    for s in np.unique(S):
        indi = S == s
        _L_sub = L[indi].mean(axis=(0, 2))
        _D_sub = D[indi]
        _C_sub = C[indi]
        assert len(np.unique(_D_sub)) == 1
        assert len(np.unique(_C_sub)) == 1

        S_sub.append(s)
        L_sub.append(_L_sub)
        D_sub.append(_D_sub[0])
        C_sub.append(_C_sub[0])

    S_sub = np.hstack(S_sub)
    L_sub = np.vstack(L_sub)
    D_sub = np.hstack(D_sub)
    C_sub = np.hstack(C_sub)
    return None, L_sub, D_sub, C_sub, S_sub
