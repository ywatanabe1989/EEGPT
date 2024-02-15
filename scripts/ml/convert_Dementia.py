#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-14 11:24:36 (ywatanabe)"

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import umap.umap_ as umap  # Ensure correct UMAP import
from catboost import CatBoostClassifier
from models.EEGPT import EEGPT
from scripts.data.database.SQLiteDataset import load_ds_dls  # SQLiteDataset
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder


def get_latent(dl):
    latent_all = []
    disease_type_all = []
    cognitive_level_all = []
    subject_all = []
    # for batch in dls["tra"]:
    for batch in dl:
        Db, Sb, Xb, _Mb = batch
        meta = ds.decode_meta(_Mb)

        Xb = Xb[..., :256]  # (64, 14, 256)

        latent = (
            model(Xb.cuda(), get_latent=True)
            # .reshape(len(latent), -1)
            .cpu()
            .detach()
            .numpy()
        )

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
    return latent_all, disease_type_all, cognitive_level_all, subject_all


class UMAP:
    """
    A class to encapsulate UMAP clustering within the mngs.ml.cluster module, with enhanced functionality.
    Allows for string labels and includes comparison with a secondary dataset.
    """

    @staticmethod
    def cluster(
        data,
        labels=None,
        supervised=False,
        title="UMAP Clustering",
        data2=None,
        data2_labels=None,
        alpha=0.3,
    ):
        """
        Performs UMAP clustering on the given data, with options for supervised or unsupervised learning,
        and visualizes the result. Optionally compares with a second dataset and handles string labels.
        Parameters:
        - data (np.ndarray): The input data for clustering.
        - labels (np.array, optional): Labels for each data point, can be strings, used in supervised mode.
        - supervised (bool, default=False): If True, performs supervised clustering using the provided labels.
        - title (str): Title for the plot.
        - data2 (np.ndarray, optional): A second set of data points to plot alongside the first for comparison.
        - data2_labels (np.array, optional): Labels for the second data set, can be strings.
        Returns:
        - fig (matplotlib.figure.Figure): The figure object for the UMAP visualization.
        - embedding (np.ndarray): The 2D embedding of the input data after UMAP reduction.
        """
        if supervised and labels is None:
            raise ValueError("Labels are required for supervised learning.")

        # Encode string labels to integers if necessary
        le = LabelEncoder()
        if labels is not None and supervised:
            labels = le.fit_transform(labels)

        umap_model = umap.UMAP()

        # Process the primary dataset
        if supervised:
            embedding = umap_model.fit_transform(data, y=labels)
        else:
            embedding = umap_model.fit_transform(data)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot the primary dataset
        if labels is not None:
            sns.scatterplot(
                x=embedding[:, 0],
                y=embedding[:, 1],
                hue=labels,
                ax=ax,
                palette="viridis",
                legend="full",
                s=50,
                alpha=alpha,
            )
        else:
            ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                s=50,
                label="Data1",
                alpha=alpha,
            )

        # Process and plot the secondary dataset if provided
        if data2 is not None:
            if data2_labels is not None and supervised:
                data2_labels = le.transform(
                    data2_labels
                )  # Transform labels for consistency
            embedding2 = umap_model.transform(
                data2
            )  # Use transform for the secondary dataset
            if data2_labels is not None:
                sns.scatterplot(
                    x=embedding2[:, 0],
                    y=embedding2[:, 1],
                    hue=data2_labels,
                    ax=ax,
                    palette="cool",
                    legend="full",
                    s=50,
                    marker="^",
                    alpha=alpha,
                )
            else:
                ax.scatter(
                    embedding2[:, 0],
                    embedding2[:, 1],
                    s=50,
                    label="Data2",
                    marker="^",
                    alpha=alpha,
                )

        ax.set_title(title)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.legend()

        return fig, embedding


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

    # Convert true labels to numeric if they are not already
    if (
        true_labels.dtype.kind in "UO"
    ):  # Check if labels are strings or objects
        le = LabelEncoder()
        true_labels = le.fit_transform(true_labels)

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


if __name__ == "__main__":
    # Configurations
    CONFIG, sys.stdout, sys.stderr, plt, cc = mngs.gen.start(
        sys,
        plt,
    )
    WINDOW_SIZE_PTS = CONFIG["WINDOW_SIZE_SEC"] * CONFIG["TGT_FS"]
    mngs.gen.fix_seeds(np=np, torch=torch)

    # Model
    model = EEGPT().cuda()
    weights_path = (
        # "./scripts/ml/pretrain_EEGPT/2024-02-11-07-05_dPXGGb4b/checkpoint.pth"
        # "./scripts/ml/pretrain_EEGPT/2024-02-11-11-43_jRsr4vFQ/checkpoint.pth"
        # "./scripts/ml/pretrain_EEGPT/2024-02-13-07-32_LbFdGgTo/checkpoint.pth"
        "./scripts/ml/pretrain_EEGPT/2024-02-13-09-39_nNwPoQxe/checkpoint.pth"  # subj-level: 0.28
    )
    x = torch.randn(32, 14, 256).cuda()
    model(x)  # to build the vq layer
    model.load_state_dict(mngs.io.load(weights_path).state_dict())

    # Loads data
    ds, dls = load_ds_dls("./config/DownStream/Dementia.yaml")

    # Converts to latent features
    # latent_features, disease_types, cognitive_levels, subjects
    l_tra, d_tra, c_tra, s_tra = get_latent(dls["tra"])
    l_val, d_val, c_val, s_val = get_latent(dls["val"])
    l_tes, d_tes, c_tes, s_tes = get_latent(dls["tes"])

    X_tra, T_tra, S_tra = l_tra, d_tra, s_tra
    X_tes, T_tes, S_tes = l_val, d_val, s_val

    # Classication
    clf = CatBoostClassifier()
    clf.fit(X_tra, T_tra)

    # Segment-level prediction
    segment_accuracy, segment_predictions = predict_segment_level(
        clf, X_tes, T_tes
    )
    print(f"Accuracy for segments: {segment_accuracy:.3f}")

    # Subject-level prediction
    true_subject_labels = (
        pd.Series(T_tes)
        .groupby(S_tes)
        .agg(lambda x: x.value_counts().idxmax())
        .values
    )
    subject_balanced_accuracy, subject_predictions = predict_subject_level(
        S_tes, clf.predict_proba(X_tes), true_subject_labels
    )
    print(
        f"Balanced Accuracy Score for subjects: {subject_balanced_accuracy:.3f}"
    )

    # UMAP Visualization
    fig, embedding = UMAP().cluster(
        X_tra,
        labels=T_tra,
        supervised=False,
        title="UMAP Clustering",
        data2=X_tes[:300],
        data2_labels=T_tes[:300],
    )
    plt.show()
