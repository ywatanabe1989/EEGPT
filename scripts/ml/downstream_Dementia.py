#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-16 22:58:21 (ywatanabe)"
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
from catboost import CatBoostClassifier
from models.EEGPT import EEGPT
from scripts.data.database.SQLiteDataset import load_ds_dls  # SQLiteDataset
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm


def get_latent(dl, windo_size, random_slice=False):
    latent_all = []
    disease_type_all = []
    cognitive_level_all = []
    subject_all = []

    start = random.randint(0, WINDOW_SIZE_PTS - 1) if random_slice else 0

    for batch in dl:
        Db, Sb, Xb, _Mb = batch
        meta = ds.decode_meta(_Mb)

        Xb = Xb[..., start : start + WINDOW_SIZE_PTS]  # (64, 14, 256)

        latent = model(Xb.cuda(), get_latent=True).cpu().detach().numpy()

        latent_all.append(latent)
        subject_all.append(Sb)
        disease_type_all.append(meta["disease_type"])
        cognitive_level_all.append(meta["cognitive_level"])

    latent_all = np.vstack(latent_all)
    latent_all = latent_all.reshape(len(latent_all), -1)
    disease_type_all = np.hstack(disease_type_all)
    cognitive_level_all = np.hstack(cognitive_level_all)
    subject_all = np.hstack(subject_all)

    assert (
        len(latent_all)
        == len(disease_type_all)
        == len(cognitive_level_all)
        == len(subject_all)
    )
    return (
        latent_all,
        disease_type_all,
        cognitive_level_all,
        subject_all,
    )


def augument_with_sliding_window(n_epochs=100):
    # Initialize lists to store results for each epoch
    l_tra_epochs, d_tra_epochs, c_tra_epochs, s_tra_epochs = [], [], [], []

    # Loop over 100 epochs to collect results
    for epoch in tqdm(range(n_epochs)):
        _l_tra, _d_tra, _c_tra, _s_tra = get_latent(
            dls["tra"], WINDOW_SIZE_PTS, random_slice=True
        )

        # Append the results of this epoch to the lists
        l_tra_epochs.append(_l_tra)
        d_tra_epochs.append(_d_tra)
        c_tra_epochs.append(_c_tra)
        s_tra_epochs.append(_s_tra)

    l_tra_epochs = np.vstack(l_tra_epochs)
    d_tra_epochs = np.hstack(s_tra_epochs)
    c_tra_epochs = np.hstack(c_tra_epochs)
    s_tra_epochs = np.hstack(s_tra_epochs)

    return l_tra_epochs, d_tra_epochs, c_tra_epochs, s_tra_epochs


class UMAP:
    """
    A class to encapsulate UMAP clustering within the mngs.ml.cluster module, with enhanced functionality.
    Allows for string labels and includes comparison with a secondary dataset.
    """

    @staticmethod
    def cluster(
        data_all,
        labels_all,
        axes_titles=None,
        supervised=False,
        title="UMAP Clustering",
        alpha=0.3,
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

        umap_model = umap.UMAP(random_state=42)

        # Process the primary dataset
        if supervised:
            _umap = umap_model.fit(data_all[0], y=labels_all[0])
            title = f"(Supervised) {title}"
        else:
            _umap = umap_model.fit(data_all[0])
            title = f"(Unsupervised) {title}"

        fig, axes = plt.subplots(
            ncols=len(data_all) + 1, sharex=True, sharey=True
        )
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
                s=50,
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
                s=50,
                alpha=alpha,
            )
            ax.set_aspect("equal")

            if axes_titles is not None:
                ax.set_title(axes_titles[ii])

        fig.suptitle(title)
        fig.supxlabel("UMAP 1")
        fig.supylabel("UMAP 2")

        # # Adjust legend
        # handles, labels = axes[0].get_legend_handles_labels()
        # fig.legend(handles, labels, loc="lower center", ncol=3)

        return fig, _umap


def predict_segment_level(classifier, features, true_labels):
    """
    Make segment-level predictions and calculate accuracy.

    Parameters:
    - classifier: The trained classifier (e.g., CatBoostClassifier).
    - features: The feature array for test segments.
    - true_labels: The true labels for the test segments.

    Returns:
    - accuracy: The accuracy of the segment-level predictions.
    - pred_class: The predicted classes for the test segments.
    """
    pred_proba = classifier.predict_proba(features)
    pred_class = np.argmax(pred_proba, axis=1)

    # # Convert true labels to numeric if they are not already
    # cvt_d = {cls: i_cls for i_cls, cls in enumerate(classifier.classes_)}
    # if (
    #     true_labels.dtype.kind in "UO"
    # ):  # Check if labels are strings or objects
    #     true_labels = [cvt_d[tl] for tl in true_labels]

    accuracy = (pred_class == true_labels).mean()
    return accuracy, pred_class


def predict_subject_level(subject_ids, pred_proba, true_labels):
    """
    Make subject-level predictions by averaging segment predictions and calculate balanced accuracy.

    Parameters:
    - subject_ids: An array of subject identifiers corresponding to each segment.
    - pred_proba: The array of predicted probabilities for each segment.
    - true_labels: The true labels for each subject.

    Returns:
    - balanced_acc: The balanced accuracy score for subject-level predictions.
    - subject_pred_class: The predicted class for each subject.
    """
    # Convert subject identifiers to a pandas Series for easy grouping
    subjects_series = pd.Series(subject_ids, name="Subject")

    # Convert predicted probabilities to a DataFrame
    pred_proba_df = pd.DataFrame(pred_proba)

    # Group by subject and calculate the mean predicted probabilities for each subject
    subject_pred_proba_mean = pred_proba_df.groupby(subjects_series).mean()

    # Determine the predicted class for each subject by finding the class with the highest mean probability
    subject_pred_class_indices = subject_pred_proba_mean.values.argmax(axis=1)

    # Ensure that true_labels are numeric
    le = LabelEncoder()
    true_labels_numeric = le.fit_transform(true_labels)

    # Calculate the balanced accuracy score
    balanced_acc = balanced_accuracy_score(
        true_labels_numeric, subject_pred_class_indices
    )

    return balanced_acc, subject_pred_class_indices


def freeze_batchnorm_stats(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


class DeterministicDropout(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, x):
        # if self.training:
        #     raise RuntimeError("This module only works in evaluation mode.")
        # else:
        if self.mask is None or self.mask.shape != x.shape:
            # Initialize or update the mask
            device = x.device
            mask = torch.ones_like(x, device=device)
            num_elements = x.nelement()
            num_drop = int(self.dropout_rate * num_elements)
            mask = mask.flatten()
            mask[:num_drop] = 0
            mask = mask.reshape(x.shape).to(device)
            self.mask = mask
        return x * self.mask


def replace_dropout_with_deterministic(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.Dropout):
            deterministic_dropout = DeterministicDropout(dropout_rate=child.p)
            setattr(model, child_name, deterministic_dropout)
        else:
            replace_dropout_with_deterministic(child)


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First hidden layer
        self.dropout1 = nn.Dropout(0.5)  # Dropout layer with 50% probability
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Second hidden layer
        self.dropout2 = nn.Dropout(
            0.5
        )  # Another dropout layer with 50% probability
        self.fc3 = nn.Linear(hidden_size, output_size)  # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Activation function for hidden layer
        x = self.dropout1(x)  # Apply dropout
        x = F.relu(self.fc2(x))  # Activation function for hidden layer
        x = self.dropout2(x)  # Apply dropout
        x = self.fc3(x)  # No activation function for output layer
        return x


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train the EEGPT model with given hyperparameters."
    )
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=1.0,
        help="Weight for the commitment loss",
    )
    parser.add_argument(
        "-b",
        "--beta",
        type=float,
        default=0.3,
        help="Weight for the entropy loss",
    )
    parser.add_argument(
        "-g",
        "--gamma",
        type=float,
        default=0.0,
        help="Weight for the prediction loss",
    )
    parser.add_argument(
        "--softmax_type",
        default="gumbel_softmax_hard",
        choices=["softmax", "gumbel_softmax_soft", "gumbel_softmax_hard"],
        help="Type of softmax to be used in the VQ layer: 'softmax', 'gumbel_softmax_soft', or 'gumbel_softmax_hard'.",
    )
    parser.add_argument(
        "--num_embeddings",
        default=4,
        help="",
    )
    parser.add_argument(
        "--embedding_dim",
        default=16,
        help="",
    )
    parser.add_argument(
        "--weight_norm",
        default=False,
        help="",
    )
    parser.add_argument(
        "--initializer",
        default="uniform",
        help="",
    )
    parser.add_argument(
        "--load_conv_and_vq",
        action="store_true",
        default=False,
        help="Load convolution and vector quantization layers and freeze them.",
    )

    args = parser.parse_args()

    print(f"\nargs:\n{args}")

    model_config = dict(
        VQ_num_embeddings=args.num_embeddings,
        VQ_embedding_dim=args.embedding_dim,
        VQ_softmax_type=args.softmax_type,
        VQ_weight_norm=args.weight_norm,
        VQ_initializer=args.initializer,
        loss_alpha=args.alpha,
        loss_beta=args.beta,
        loss_gamma=args.gamma,
        load_conv_and_vq=args.load_conv_and_vq,
    )

    # Configurations
    CONFIG, sys.stdout, sys.stderr, plt, cc = mngs.gen.start(
        sys,
        plt,
    )
    WINDOW_SIZE_PTS = CONFIG["WINDOW_SIZE_SEC"] * CONFIG["TGT_FS"]
    mngs.gen.fix_seeds(np=np, torch=torch)

    # Model
    model = EEGPT(model_config).cuda()
    weights_path = (
        # "./scripts/ml/pretrain_EEGPT/2024-02-11-07-05_dPXGGb4b/checkpoint.pth"
        # "./scripts/ml/pretrain_EEGPT/2024-02-11-11-43_jRsr4vFQ/checkpoint.pth"
        # "./scripts/ml/pretrain_EEGPT/2024-02-13-07-32_LbFdGgTo/checkpoint.pth"
        # "./scripts/ml/pretrain_EEGPT/2024-02-13-09-39_nNwPoQxe/checkpoint.pth"  # subj-level: 0.28
        "./scripts/ml/pretrain_EEGPT/2024-02-16-11-44_prtSjkMu/checkpoint.pth"
    )

    model.load_state_dict(mngs.io.load(weights_path).state_dict())
    # replace_dropout_with_deterministic(model)
    freeze_batchnorm_stats(model)
    # model.eval()
    # mngs.ml.utils.check_params(model)

    # model = model.cuda()

    # Loads data
    ds, dls = load_ds_dls("./config/DownStream/Dementia.yaml")

    # Converts to latent features
    # latent_features, disease_types, cognitive_levels, subjects
    WINDOW_SIZE_PTS = int(CONFIG["WINDOW_SIZE_SEC"] * CONFIG["TGT_FS"])
    l_tra, d_tra, c_tra, s_tra = get_latent(
        dls["tra"], WINDOW_SIZE_PTS, random_slice=False
    )
    l_val, d_val, c_val, s_val = get_latent(
        dls["val"], WINDOW_SIZE_PTS, random_slice=False
    )
    l_tes, d_tes, c_tes, s_tes = get_latent(
        dls["tes"], WINDOW_SIZE_PTS, random_slice=False
    )

    X_tra, T_tra, S_tra = l_tra, d_tra, s_tra
    X_val, T_val, S_val = l_val, d_val, s_val
    X_tes, T_tes, S_tes = l_tes, d_tes, s_tes

    l_tra, d_tra, c_tra, s_tra = augument_with_sliding_window()
    X_tra, T_tra, S_tra = l_tra, d_tra, s_tra

    # # UMAP Visualization
    # for label_type in ["subject", "disease_type", "cognitive_level"]:
    #     label_tra, label_val, label_tes = {
    #         "subject": (s_tra, s_val, s_tes),
    #         "disease_type": (d_tra, d_val, d_tes),
    #         "cognitive_level": (c_tra, c_val, c_tes),
    #     }[label_type]

    #     for supervised in [False, True]:
    #         fig, _ = UMAP().cluster(
    #             [X_tra, X_val, X_tes],
    #             [label_tra, label_val, label_tes],
    #             axes_titles=["Training", "Validation", "Test"],
    #             supervised=supervised,
    #             alpha=0.5,
    #             title=f"UMAP Clustering (Labeled by {label_type})",
    #         )
    #         supervised_str = "supervised" if supervised else "unsupervised"

    #         mngs.io.save(
    #             fig, CONFIG["SDIR"] + f"UMAP/{supervised_str}/{label_type}.png"
    #         )

    # Classication
    umap_model = umap.UMAP(
        n_neighbors=15, n_components=5, metric="euclidean", random_state=42
    )
    umap_model.fit(X_tra)
    X_tra_umap = umap_model.transform(X_tra)
    X_val_umap = umap_model.transform(X_val)
    X_tes_umap = umap_model.transform(X_tes)

    # clf = CatBoostClassifier()
    clf = MLPClassifier(
        hidden_layer_sizes=(10, 10), max_iter=300, random_state=42
    )
    # clf = SVC(probability=True, class_weight='balanced') # [REVISED]

    # # Normalize the data
    # scaler_tra = StandardScaler()
    # X_tra_last = scaler_tra.fit_transform(X_tra_umap)

    # scaler_val = StandardScaler()
    # X_val_last = scaler_val.fit_transform(X_val_umap)

    # scaler_tes = StandardScaler()
    # X_tes_last = scaler_tes.fit_transform(X_tes_umap)

    X_tra_last = X_tra_umap
    X_val_last = X_val_umap
    X_tes_last = X_tes_umap

    clf.fit(X_tra_last, T_tra)

    # Convert true labels to numeric if they are not already
    cvt_d = {cls: i_cls for i_cls, cls in enumerate(clf.classes_)}
    T_val_int = [cvt_d[tt] for tt in T_val]
    T_tes_int = [cvt_d[tt] for tt in T_tes]

    # Segment-level prediction
    segment_accuracy, segment_predictions = predict_segment_level(
        clf, X_val_last, T_val_int
    )
    print(f"Accuracy for segments: {segment_accuracy:.3f}")

    # Subject-level prediction
    T_val_int_subject = (
        pd.Series(T_val_int)
        .groupby(S_val)
        .agg(lambda x: x.value_counts().idxmax())
        .values
    )
    subject_balanced_accuracy, subject_predictions = predict_subject_level(
        S_val, clf.predict_proba(X_val_last), T_val_int_subject
    )
    print(
        f"Balanced Accuracy Score for subjects: {subject_balanced_accuracy:.3f}"
    )

    # Confusion Matrix
    cm = confusion_matrix(T_val_int_subject, subject_predictions)
    mngs.ml.plt.confusion_matrix(plt, cm)
    plt.show()
    # np.mean(T_val_int_subject == subject_predictions)
