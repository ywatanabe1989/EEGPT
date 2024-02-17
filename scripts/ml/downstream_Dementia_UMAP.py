#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-17 20:24:16 (ywatanabe)"
import random
import sys

import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import umap.umap_ as umap
import xgboost as xgb
from catboost import CatBoostClassifier
from models.EEGPT import EEGPT
from scripts.data.database.SQLiteDataset import load_ds_dls  # SQLiteDataset
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm


def get_latent(dl, window_size):
    X_all = []
    L_all = []
    Q_all = []
    D_all = []
    C_all = []
    S_all = []

    EEGPT_model.eval()
    for batch in dl:
        Db, Sb, Xb, _Mb = batch
        meta = ds.decode_meta(_Mb)

        Xb = torch.rand(Xb.shape)  # fixme

        Lb = EEGPT_model(Xb.cuda(), get_latent=True).cpu().detach().numpy()

        Qb = EEGPT_model.assignment_probs

        X_all.append(Xb.cpu().detach().numpy())
        L_all.append(Lb)
        Q_all.append(Qb.cpu().detach().numpy())
        S_all.append(Sb)
        D_all.append(meta["disease_type"])
        C_all.append(meta["cognitive_level"])

    X_all = np.vstack(X_all)
    L_all = np.vstack(L_all)
    Q_all = np.vstack(Q_all)
    D_all = np.hstack(D_all)
    C_all = np.hstack(C_all)
    S_all = np.hstack(S_all)

    assert (
        len(X_all)
        == len(L_all)
        == len(Q_all)
        == len(D_all)
        == len(C_all)
        == len(S_all)
    )

    return (
        X_all,
        L_all,
        Q_all,
        D_all,
        C_all,
        S_all,
    )


def augument_with_sliding_window(dl, n_epochs=10):
    # Initialize lists to store results for each epoch
    X_all, L_all, Q_all, D_all, C_all, S_all = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    # Loop over 100 epochs to collect results
    for epoch in tqdm(range(n_epochs)):
        (x, l, q, d, c, s) = get_latent(dl, WINDOW_SIZE_PTS)

        # Append the results of this epoch to the lists
        X_all.append(x)
        L_all.append(l)
        Q_all.append(q)
        D_all.append(d)
        C_all.append(c)
        S_all.append(s)

    X_all = np.vstack(X_all)
    L_all = np.vstack(L_all)
    Q_all = np.vstack(Q_all)
    D_all = np.hstack(D_all)
    C_all = np.hstack(C_all)
    S_all = np.hstack(S_all)

    return X_all, L_all, Q_all, D_all, C_all, S_all


def to_sub_mean(_, L, Q, D, C, S):
    S_sub = []
    L_sub = []
    Q_sub = []
    D_sub = []
    C_sub = []
    for s in np.unique(S):
        indi = S == s
        _L_sub = L[indi].mean(axis=(0, 2))
        _Q_sub = Q[indi].mean(axis=(0, 1))
        _D_sub = D[indi]
        _C_sub = C[indi]
        assert len(np.unique(_D_sub)) == 1
        assert len(np.unique(_C_sub)) == 1

        S_sub.append(s)
        L_sub.append(_L_sub)
        Q_sub.append(_Q_sub)
        D_sub.append(_D_sub[0])
        C_sub.append(_C_sub[0])

    S_sub = np.hstack(S_sub)
    L_sub = np.vstack(L_sub)
    Q_sub = np.vstack(Q_sub)
    D_sub = np.hstack(D_sub)
    C_sub = np.hstack(C_sub)
    return None, L_sub, Q_sub, D_sub, C_sub, S_sub


def visualize_by_umap(
    data_all,
    labels_all,
    axes_titles=None,
    supervised=False,
    title="UMAP Clustering",
    alpha=0.1,
    s=3,
):

    assert len(data_all) == len(labels_all)

    if isinstance(data_all, list):
        data_all = list(data_all)
        labels_all = list(labels_all)

    le = LabelEncoder()
    # Store original labels
    labels_all_orig = [np.array(labels) for labels in labels_all]
    if labels_all is not None:
        labels_uq = np.unique(np.hstack(labels_all))
        le.fit(labels_uq)
        labels_all = [le.transform(labels) for labels in labels_all]

    umap_EEGPT_model = umap.UMAP(random_state=42)

    # Process the primary dataset
    if supervised:
        _umap = umap_EEGPT_model.fit(data_all[0], y=labels_all[0])
        title = f"(Supervised) {title}"
    else:
        _umap = umap_EEGPT_model.fit(data_all[0])
        title = f"(Unsupervised) {title}"

    fig, axes = plt.subplots(ncols=len(data_all) + 1, sharex=True, sharey=True)
    for ii, (data, labels, labels_orig) in enumerate(
        zip(data_all, labels_all, labels_all_orig)
    ):
        embedding = _umap.transform(data)
        ax = axes[ii + 1]

        # Superimposed
        axes[0].set_title("Superimposed")
        axes[0].set_aspect("equal")
        sns.scatterplot(
            x=embedding[:, 0],
            y=embedding[:, 1],
            hue=labels_orig,
            ax=axes[0],
            palette="viridis",
            legend="full" if ii == 0 else False,
            s=s,
            alpha=alpha,
        )

        # Each data
        sns.scatterplot(
            x=embedding[:, 0],
            y=embedding[:, 1],
            hue=labels_orig,
            ax=ax,
            palette="viridis",
            legend=False,
            s=s,
            alpha=alpha,
        )
        ax.set_aspect("equal")

        if axes_titles is not None:
            ax.set_title(axes_titles[ii])

    # Place the legend outside of the last axes
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        title="Legend",
    )

    # Adjust the layout to make room for the legend
    plt.tight_layout(
        rect=[0, 0, 0.85, 1]
    )  # Adjust the rect parameter as needed

    fig.suptitle(title)
    fig.supxlabel("UMAP 1")
    fig.supylabel("UMAP 2")

    return fig, _umap


def vis_umap_all():
    # UMAP Visualization
    for label_type in ["subject", "disease_type", "cognitive_level"]:
        label_tra, label_val, label_tes = {
            "subject": (S_tra, S_val, S_tes),
            "disease_type": (D_tra, D_val, D_tes),
            "cognitive_level": (C_tra, C_val, C_tes),
        }[label_type]

        for supervised in [False, True]:
            fig, _ = visualize_by_umap(
                [
                    L_tra.mean(axis=-1),
                    L_val.mean(axis=-1),
                    L_tes.mean(axis=-1),
                ],
                [label_tra, label_val, label_tes],
                axes_titles=["Training", "Validation", "Test"],
                supervised=supervised,
                alpha=0.5,
                title=f"UMAP Clustering (Labeled by {label_type})",
            )
            supervised_str = "supervised" if supervised else "unsupervised"

            mngs.io.save(
                fig, CONFIG["SDIR"] + f"UMAP/{supervised_str}/{label_type}.png"
            )


def predict_segment_level(classifier, features, true_labels):
    pred_proba = classifier.predict_proba(features)
    pred_class = np.argmax(pred_proba, axis=1)
    balanced_accuracy = balanced_accuracy_score(pred_class, true_labels)
    print(f"\nBalanced Accuracy for segments: {balanced_accuracy:.3f}")
    return balanced_accuracy, pred_class


def predict_subject_level(clf, X, T, S):
    pred_proba_seg = clf.predict_proba(X)

    pred_proba_sub = []
    T_sub = []

    for sub in np.unique(S):
        indi = S == sub
        pred_proba_sub.append(pred_proba_seg[indi].mean(axis=0))
        assert np.all(T[indi] == T[indi][0])
        T_sub.append(T[0])

    pred_proba_sub = np.array(pred_proba_sub)
    T_sub = np.hstack(T_sub)
    pred_cls_sub = pred_proba_sub.argmax(axis=-1)
    balanced_accuracy = balanced_accuracy_score(pred_cls_sub, T_sub)
    print(f"\nBalanced Accuracy Score for subjects: {balanced_accuracy:.3f}")
    return balanced_accuracy, pred_cls_sub


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train the EEGPT EEGPT_model with given hyperparameters."
    )
    parser.add_argument(
        "--weights_path",
        default="./scripts/ml/pretrain_EEGPT/2024-02-17-08-15_UmciKUpN/checkpoint.pth",
        help="Load convolution and vector quantization layers and freeze them.",
    )
    args = parser.parse_args()
    print(f"\nargs:\n{args}")

    EEGPT_model_config = dict(
        VQ_num_embeddings=4,
        VQ_embedding_dim=16,
        VQ_softmax_type="gumbel_softmax_hard",
        VQ_weight_norm=False,
        VQ_initializer="uniform",
        loss_commit_coef=0.0,
        loss_entropy_coef=0.0,
        loss_pred_coef=0.0,
        weights_path=args.weights_path,
    )

    # Configurations
    CONFIG, sys.stdout, sys.stderr, plt, cc = mngs.gen.start(
        sys,
        plt,
        np=np,
        torch=torch,
    )
    plt, cc = mngs.plt.configure_mpl(plt, fig_scale=3)
    WINDOW_SIZE_PTS = CONFIG["WINDOW_SIZE_SEC"] * CONFIG["TGT_FS"]
    # mngs.gen.fix_seeds(np=np, torch=torch)

    # EEGPT_Model
    EEGPT_model = EEGPT(EEGPT_model_config).cuda().eval()
    # Xb = torch.rand(64, 14, 256).cuda()
    # Lb = EEGPT_model(Xb.cuda(), get_latent=True).cpu().detach().numpy()
    # Lb[0,:,0]
    # Lb[0,:,1]
    # freeze_batchnorm_stats(EEGPT_model)

    # Loads data
    ds, dls = load_ds_dls("./config/DownStream/Dementia.yaml")

    # Converts to latent features
    # N_EPOCHS = 10

    X_tra, L_tra, Q_tra, D_tra, C_tra, S_tra = augument_with_sliding_window(
        dls["tra"],
        n_epochs=10,
    )
    X_val, L_val, Q_val, D_val, C_val, S_val = augument_with_sliding_window(
        dls["val"],
        n_epochs=2,
    )
    X_tes, L_tes, Q_tes, D_tes, C_tes, S_tes = augument_with_sliding_window(
        dls["tes"],
        n_epochs=2,
    )
    Xs_tra, Ls_tra, Qs_tra, Ds_tra, Cs_tra, Ss_tra = to_sub_mean(
        X_tra, L_tra, Q_tra, D_tra, C_tra, S_tra
    )
    Xs_val, Ls_val, Qs_val, Ds_val, Cs_val, Ss_val = to_sub_mean(
        X_val, L_val, Q_val, D_val, C_val, S_val
    )
    Xs_tes, Ls_tes, Qs_tes, Ds_tes, Cs_tes, Ss_tes = to_sub_mean(
        X_tes, L_tes, Q_tes, D_tes, C_tes, S_tes
    )

    sns.heatmap(L_tra[0])
    plt.show()

    # # Visualize data with UMAP
    # vis_umap_all()

    # L reshaped
    X_tra = L_tra.reshape(len(L_tra), -1)
    X_val = L_val.reshape(len(L_val), -1)
    X_tes = L_tes.reshape(len(L_tes), -1)

    # # L mean
    # X_tra = L_tra.mean(axis=-1)
    # X_val = L_val.mean(axis=-1)
    # X_tes = L_tes.mean(axis=-1)

    # # Q reshaped
    # X_tra = Q_tra.reshape(len(Q_tra), -1)
    # X_val = Q_val.reshape(len(Q_val), -1)
    # X_tes = Q_tes.reshape(len(Q_tes), -1)

    # # Q argmax
    # X_tra = Q_tra.argmax(axis=-1)
    # X_val = Q_val.argmax(axis=-1)
    # X_tes = Q_tes.argmax(axis=-1)

    # X_tra.sort(axis=-1)
    # X_val.sort(axis=-1)
    # X_tes.sort(axis=-1)

    T_tra = D_tra
    T_val = D_val
    T_tes = D_tes

    DICT = {tt: ii for ii, tt in enumerate(np.unique(T_tra))}

    T_tra = np.hstack([DICT[tt] for tt in T_tra])
    T_val = np.hstack([DICT[tt] for tt in T_val])
    T_tes = np.hstack([DICT[tt] for tt in T_tes])

    # # Classification
    # clf = xgb.XGBClassifier(
    #     objective="multi:softprob",  # Use 'binary:logistic' for binary classification
    #     eval_metric="mlogloss",  # Use 'logloss' for binary classification
    #     # use_label_encoder=False,  # To avoid a deprecation warning from XGBoost
    #     early_stopping_rounds=10,  # Stop if validation score doesn't improve for 10 rounds
    #     random_state=42,
    # )

    # clf.fit(
    #     X_tra,
    #     T_tra,
    #     eval_set=[(X_val, T_val)],
    #     verbose=True,
    # )

    clf = CatBoostClassifier()
    # clf = MLPClassifier(
    #     hidden_layer_sizes=(3, 3), max_iter=300, random_state=42
    # )

    # X_tra = mngs.gen.to_z(X_tra, axis=-1)
    # X_val = mngs.gen.to_z(X_val, axis=-1)
    # X_tes = mngs.gen.to_z(X_tes, axis=-1)

    # clf = SVC(probability=True, class_weight="balanced")
    clf.fit(
        X_tra,
        T_tra,
    )

    # Segment-level prediction
    segment_accuracy, segment_predictions = predict_segment_level(
        clf,
        X_val,
        T_val,
    )
    # Subject-level prediction
    subject_balanced_accuracy, subject_predictions = predict_subject_level(
        clf, X_val, T_val, S_val
    )

    segment_accuracy, segment_predictions = predict_segment_level(
        clf,
        X_tes,
        T_tes,
    )

    # Subject-level prediction
    subject_balanced_accuracy, subject_predictions = predict_subject_level(
        clf, X_tes, T_tes, S_tes
    )
