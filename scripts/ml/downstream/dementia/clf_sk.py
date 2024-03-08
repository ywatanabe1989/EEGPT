#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-26 08:22:01 (ywatanabe)"
import sys

import matplotlib.pyplot as plt
import mngs
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from catboost import CatBoostClassifier
from models.CNN1D import CNN1D
from models.EEGPT import load_trained_EEGPT
from scripts.ml.downstream.Dementia.utils import load_transfered_data
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset


def predict_segment_level(clf, X, T):
    pred_proba = clf.predict_proba(X)
    pred_class = np.argmax(pred_proba, axis=-1)
    balanced_accuracy = balanced_accuracy_score(pred_class, T)
    print(f"\nBalanced Accuracy for segments: {balanced_accuracy:.3f}")


def predict_subject_level(clf, X, T, S):
    pred_proba_seg = clf.predict_proba(X)

    pred_proba_sub = []
    T_sub = []

    for sub in np.unique(S):
        indi = S == sub
        pred_proba_sub.append(pred_proba_seg[indi].mean(axis=0))
        assert np.all(T[indi] == T[indi][0])
        T_sub.append(T[indi][0])

    pred_proba_sub = np.array(pred_proba_sub)
    T_sub = np.hstack(T_sub)
    pred_cls_sub = pred_proba_sub.argmax(axis=-1)
    balanced_accuracy = balanced_accuracy_score(pred_cls_sub, T_sub)
    print(f"\nBalanced Accuracy Score for subjects: {balanced_accuracy:.3f}")


def to_stats(X_tra):
    return np.stack(
        [
            mngs.dsp.feature_extractors.mean(X_tra),
            mngs.dsp.feature_extractors.std(X_tra),
            mngs.dsp.feature_extractors.skewness(X_tra),
            mngs.dsp.feature_extractors.kurtosis(X_tra),
            mngs.dsp.feature_extractors.q25(X_tra),
            mngs.dsp.feature_extractors.q75(X_tra),
        ],
        axis=-1,
    ).squeeze()


class PyTorchMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        input_size,
        hidden_layer_sizes=(100,),
        dropout=0.5,
        max_iter=300,
        batch_size=64,
        learning_rate=1e-3,
        device=None,
    ):
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout = dropout
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and device is None else device
        )
        self.model_ = self._build_model().to(self.device)

    def _build_model(self):
        layers = []
        input_size = self.input_size
        for hidden_size in self.hidden_layer_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            input_size = hidden_size
        layers.append(
            nn.Linear(input_size, 10)
        )  # Assuming 10 classes for the output layer
        return nn.Sequential(*layers)

    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.int64).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        self.model_.train()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.max_iter):
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = self.model_(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, X):
        self.model_.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            outputs = self.model_(X_tensor)
            probabilities = nn.functional.softmax(outputs, dim=1)
        return (
            probabilities.cpu().numpy()
        )  # Move the tensor back to CPU for compatibility with numpy

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class PyTorchCNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        input_shape,
        num_classes,
        dropout=0.5,
        max_iter=300,
        batch_size=64,
        learning_rate=1e-3,
        device=None,
    ):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout = dropout
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and device is None else device
        )
        self.model_ = self._build_model().to(self.device)

    def _build_model(self):
        model = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_shape[0],
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Dropout(self.dropout),
            nn.Linear(
                64 * (self.input_shape[1] // 4) * (self.input_shape[2] // 4),
                128,
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, self.num_classes),
        )
        return model

    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.int64).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        self.model_.train()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.max_iter):
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = self.model_(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, X):
        self.model_.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            outputs = self.model_(X_tensor)
            probabilities = nn.functional.softmax(outputs, dim=1)
        return probabilities.cpu().numpy()

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train the EEGPT EEGPT_model with given hyperparameters."
    )
    parser.add_argument(
        "--ID",
        # default="2024Y-02M-24D-17h46m03s_jk5y", # 0.25
        # default="2024Y-02M-24D-23h41m51s_wRO0",  # ??
        # default="2024Y-02M-24D-17h46m03s_jXr9", # ??
        # default="2024Y-02M-25D-04h32m38s_Ceqd", # 0.25
        # default="2024Y-02M-24D-23h41m51s_Lh0X",
        # default="2024Y-02M-24D-23h41m51s_Lh0X",
        # default="FINISHED/2024Y-02M-25D-12h27m14s_8agi",
        # default="FINISHED/2024Y-02M-25D-12h27m14s_tcNR",
        default="FINISHED/2024Y-02M-25D-12h27m14s_9zs9",
        # defualt="FINISHED/2024Y-02M-25D-12h27m14s_TtLn",
        help="ID for the pretrained model",
    )
    args = parser.parse_args()
    print(f"\nargs:\n{args}")

    # Configurations
    CONFIG, sys.stdout, sys.stderr, plt, cc = mngs.gen.start(
        sys,
        plt,
        np=np,
        torch=torch,
    )
    plt, cc = mngs.plt.configure_mpl(plt, fig_scale=3)

    # EEGPT_Model
    EEGPT_model = load_trained_EEGPT(args.ID)

    # Loads data
    data = load_transfered_data(EEGPT_model)

    # X
    X_str = "L"

    # T
    T_str = "D"
    le = LabelEncoder()
    le.fit(data[f"{T_str}_tra"])

    # Classifier
    # clf = xgb.XGBClassifier(
    #     # early_stopping_rounds=10,  # Stop if validation score doesn't improve for 10 rounds
    #     random_state=42,
    # )
    # clf = CatBoostClassifier()
    # clf = MLPClassifier(
    #     hidden_layer_sizes=(100, 100), max_iter=300, random_state=42, early_stopping=True,
    # )
    # clf = PyTorchMLPClassifier(
    #     input_size=6, hidden_layer_sizes=(100, 100), dropout=0.75, max_iter=300
    # )

    # In [148]: data["L_tra"].shape
    # Out[148]: (67840, 16, 7)

    # Training
    split = "tra"
    X_tra = to_stats(data[f"{X_str}_{split}"][..., -1])
    # X_tra = data[f"{X_str}_{split}"][..., -1]
    T_tra = le.transform(data[f"{T_str}_{split}"])
    clf.fit(
        X_tra,
        T_tra,
    )

    # Evaluation
    for split in ["tra", "val", "tes"]:
        # X = data[f"{X_str}_{split}"][..., -1]
        X = to_stats(data[f"{X_str}_{split}"][..., -1])
        T = le.transform(data[f"{T_str}_{split}"])
        S = data[f"S_{split}"]
        predict_segment_level(clf, X, T)
        predict_subject_level(clf, X, T, S)
