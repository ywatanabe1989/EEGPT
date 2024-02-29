#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-26 19:43:42 (ywatanabe)"

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.EEGPT import load_trained_EEGPT
from torchinfo import summary


class CNN1D(nn.Module):
    def __init__(self, num_classes, num_channels=16, EEGPT_ID=None):
        """
        Initializes the 1D CNN model.

        Parameters:
        - num_channels (int): Number of input channels.
        - num_classes (int): Number of output classes.

        Returns:
        - None
        """
        super(CNN1D, self).__init__()

        self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forward pass of the 1D CNN.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, num_channels, seq_len).

        Returns:
        - torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """

        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.relu(self.fc2(x)))
        x = self.fc3(x)
        # x = self.softmax(x)
        return x


if __name__ == "__main__":
    # from scripts.ml.parse_model_configs import parse_model_configs

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Demo data
    bs, n_chs, seq_len = 128, 16, 7
    # bs, n_chs, seq_len = 128, 14, 256
    x = torch.rand(bs, n_chs, seq_len).float().to(device)
    n_classes = 10

    # Forwarding
    model = CNN1D(
        n_classes, EEGPT_ID="FINISHED/2024Y-02M-25D-12h27m14s_9zs9"
    ).to(device)
    y = model(x)

    # Print results
    print(summary(model, x.shape))
    print(y.shape)
    print(y.mean(), y.std())

    y.sum().backward()
