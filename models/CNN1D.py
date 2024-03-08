#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-03-01 16:47:12 (ywatanabe)"

import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.EEGPT import load_trained_EEGPT
from torchinfo import summary

matplotlib.use("Agg")


class CNN1D(nn.Module):
    def __init__(self, num_classes, num_channels):
        super(CNN1D, self).__init__()

        self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)  # Batch normalization
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)  # Batch normalization
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        n_fc_in = 256
        # n_fc_in = 128 * (seq_len // 4)  # Adjusted based on the pooling layers
        self.fc1 = nn.Linear(n_fc_in, int(n_fc_in / 2))
        self.dropout1 = nn.Dropout(0.95)  # Reduced dropout rate
        self.fc2 = nn.Linear(int(n_fc_in / 2), int(n_fc_in / 4))
        self.dropout2 = nn.Dropout(0.95)  # Reduced dropout rate
        self.fc3 = nn.Linear(int(n_fc_in / 4), num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.pool(x)
        x = self.bn2(self.relu(self.conv2(x)))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


# class CNN1D(nn.Module):
#     def __init__(self, num_classes, num_channels, EEGPT_ID=None):
#         """
#         Initializes the 1D CNN model.

#         Parameters:
#         - num_channels (int): Number of input channels.
#         - num_classes (int): Number of output classes.

#         Returns:
#         - None
#         """
#         super(CNN1D, self).__init__()

#         self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
#         self.flatten = nn.Flatten()

#         n_fc_in = 256
#         self.fc1 = nn.Linear(n_fc_in, int(n_fc_in / 2))
#         self.dropout1 = nn.Dropout(0.85)
#         self.fc2 = nn.Linear(int(n_fc_in / 2), int(n_fc_in / 4))
#         self.dropout2 = nn.Dropout(0.85)
#         self.fc3 = nn.Linear(int(n_fc_in / 4), num_classes)
#         self.relu = nn.ReLU()
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         """
#         Forward pass of the 1D CNN.

#         Parameters:
#         - x (torch.Tensor): Input tensor of shape (batch_size, num_channels, seq_len).

#         Returns:
#         - torch.Tensor: Output tensor of shape (batch_size, num_classes).
#         """

#         x = self.relu(self.conv1(x))
#         x = self.pool(x)
#         x = self.relu(self.conv2(x))
#         x = self.pool(x)
#         x = self.flatten(x)
#         x = self.dropout1(self.relu(self.fc1(x)))
#         x = self.dropout2(self.relu(self.fc2(x)))
#         x = self.fc3(x)
#         # x = self.softmax(x)
#         return x


if __name__ == "__main__":
    # from scripts.ml.parse_model_configs import parse_model_configs

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Demo data
    bs, n_chs, seq_len = 128, 64, 8
    x = torch.rand(bs, n_chs, seq_len).float().to(device)
    n_classes = 4

    # Forwarding
    model = CNN1D(
        n_classes, n_chs, EEGPT_ID="FINISHED/2024Y-02M-25D-12h27m14s_9zs9"
    ).to(device)
    y = model(x)

    # Print results
    print(summary(model, x.shape))
    print(y.shape)
    print(y.mean(), y.std())

    y.sum().backward()
