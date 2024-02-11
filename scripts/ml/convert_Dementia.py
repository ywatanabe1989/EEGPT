#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-12 00:09:51 (ywatanabe)"

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import umap.umap_ as umap  # Ensure correct UMAP import
from models.EEGPT import EEGPT
from scripts.data.database.SQLiteDataset import load_ds_dls  # SQLiteDataset
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
            )
        else:
            ax.scatter(embedding[:, 0], embedding[:, 1], s=50, label="Data1")

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
                )
            else:
                ax.scatter(
                    embedding2[:, 0],
                    embedding2[:, 1],
                    s=50,
                    label="Data2",
                    marker="^",
                )

        ax.set_title(title)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.legend()

        return fig, embedding


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder


class CustomPCA:
    """
    A class to encapsulate PCA dimensionality reduction within the mngs.ml.cluster module, with enhanced functionality.
    Allows for string labels and includes comparison with a secondary dataset.
    """

    @staticmethod
    def reduce(
        data,
        n_components=2,
        labels=None,
        title="PCA Dimensionality Reduction",
        data2=None,
        data2_labels=None,
    ):
        """
        Performs PCA dimensionality reduction on the given data and visualizes the result.
        Optionally compares with a second dataset and handles string labels.

        Parameters:
        - data (np.ndarray): The input data for dimensionality reduction.
        - n_components (int): Number of components to keep.
        - labels (np.array, optional): Labels for each data point, can be strings.
        - title (str): Title for the plot.
        - data2 (np.ndarray, optional): A second set of data points to plot alongside the first for comparison.
        - data2_labels (np.array, optional): Labels for the second data set, can be strings.

        Returns:
        - fig (matplotlib.figure.Figure): The figure object for the PCA visualization.
        - pca_result (np.ndarray): The reduced data after PCA.
        """
        pca_model = PCA(n_components=n_components)

        # Fit and transform the primary dataset
        pca_result = pca_model.fit_transform(data)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot the primary dataset
        if labels is not None:
            sns.scatterplot(
                x=pca_result[:, 0],
                y=pca_result[:, 1],
                hue=labels,
                ax=ax,
                palette="viridis",
                legend="full",
                s=50,
            )
        else:
            ax.scatter(pca_result[:, 0], pca_result[:, 1], s=50, label="Data1")

        # Process and plot the secondary dataset if provided
        if data2 is not None:
            pca_result2 = pca_model.transform(
                data2
            )  # Transform the secondary dataset
            if data2_labels is not None:
                sns.scatterplot(
                    x=pca_result2[:, 0],
                    y=pca_result2[:, 1],
                    hue=data2_labels,
                    ax=ax,
                    palette="cool",
                    legend="full",
                    s=50,
                    marker="^",
                )
            else:
                ax.scatter(
                    pca_result2[:, 0],
                    pca_result2[:, 1],
                    s=50,
                    label="Data2",
                    marker="^",
                )

        ax.set_title(title)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.legend()

        return fig, pca_result


# Configurations
CONFIG, sys.stdout, sys.stderr, plt, cc = mngs.gen.start(
    sys,
    plt,
    # os=os,
    # random=random,
    # torch=torch,
    # np=np,
)
WINDOW_SIZE_PTS = CONFIG["WINDOW_SIZE_SEC"] * CONFIG["TGT_FS"]

if __name__ == "__main__":
    model = EEGPT().cuda()

    # Loads model
    weights_path = (
        # "./scripts/ml/pretrain_EEGPT/2024-02-11-07-05_dPXGGb4b/checkpoint.pth"
        "./scripts/ml/pretrain_EEGPT/2024-02-11-11-43_jRsr4vFQ/checkpoint.pth"
    )
    x = torch.randn(32, 14, 256).cuda()
    model(x)  # to build the vq layer
    model.load_state_dict(mngs.io.load(weights_path).state_dict())

    # Loads data
    ds, dls = load_ds_dls("./config/DownStream/Dementia.yaml")

    # Converts to latent features
    l_tra, d_tra, c_tra, s_tra = get_latent(dls["tra"])
    l_val, d_val, c_val, s_val = get_latent(dls["val"])
    l_tes, d_tes, c_tes, s_tes = get_latent(dls["tes"])

    # UMAP = mngs.ml.clustering.UMAP()
    # UMAP = UMAP()
    fig, embedding = UMAP().cluster(
        l_tra,
        labels=c_tra,
        supervised=False,
        title="UMAP Clustering",
        data2=l_tes,
        data2_labels=c_tes,
    )
    plt.show()

    fig, embedding = CustomPCA().reduce(
        l_tra,
        labels=c_tra,
        n_components=4,
        title="PCA Dimensionality Reduction",
        data2=l_tes,
        data2_labels=c_tes,
    )
    plt.show()

    from catboost import CatBoostClassifier

    # Initialize the CatBoostClassifier
    clf = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        eval_metric="Accuracy",  # You can choose other metrics like 'AUC', 'CrossEntropy', etc.
        random_seed=42,
        verbose=200,  # Output the log every 200 iterations
    )

    # Now you can train the classifier using the fit method
    clf.fit(l_tra, d_tra)

    # After training, you can make predictions
    predictions = clf.predict_proba(l_tes)
    df = pd.DataFrame(
        index=s_tes,
        data=predictions,
    )
    cls_mean_all = []
    cls_all = []
    for sub in df.index:
        cls_mean = df[df.index == sub].mean(axis=0)
        cls = df[df.index == sub].iloc[0]

        cls_mean_all.append(cls_mean)
        cls_all.append(cls)

    pred_cls = np.argmax(pd.DataFrame(cls_mean_all), axis=1)
    cls_all

    sub = df.index[0]
    balanced_accuracy_score(predictions, d_tes)

    # from scipy.spatial.distance import cosine

    # for _ in range(10):
    #     i1 = random.randint(0, len(l))
    #     i2 = random.randint(0, len(l))
    #     # print(np.corrcoef(l[i1], l[i2])[0, 1])
    #     print("\n" + "-" * 40)
    #     print(meta.iloc[i1])
    #     print(meta.iloc[i2])
    #     print(cosine(l[i1], l[i2]))  # [0, 1])
    #     print("\n" + "-" * 40)

    # # 0.9563446985552332
    # # 0.9528256462576895
    # # 0.9579907183575023
    # # 0.9547480838781267
    # # 0.9534788045611708
    # # 0.9545304281407437
    # # 0.9542579286561749
    # # 0.9572349185429292
    # # 0.9565238602300236
    # # 0.9514539139266924
