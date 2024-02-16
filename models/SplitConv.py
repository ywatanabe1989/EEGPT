#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-17 02:36:21 (ywatanabe)"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class SplitConv(nn.Module):
    """Applies SegmentConv2D for each part of segment in a shared manner"""

    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        self.SegmentConv2D = SegmentConv2D()  # [-1, 14, 32] -> [-1, 16]

    def forward(self, x):
        xs = [
            x[
                ...,
                ii
                * self.model_config["window_size"] : (ii + 1)
                * self.model_config["window_size"],
            ]
            for ii in range(x.shape[-1] // self.model_config["window_size"])
        ]
        xs = [self.SegmentConv2D(x.unsqueeze(1)) for x in xs]
        return xs


class SegmentConv2D(nn.Module):
    """
    14 channels x 32 time steps (= 250 ms at 128 Hz)-> 16 dimensional vector

    Example:

        bs, n_chs, seq_len = 128, 14, 32
        x = torch.rand(bs, 1, n_chs, seq_len).float()
        model = SegmentConv2D()
        y = model(x)  # (-1, 16)
    """

    def __init__(self):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(16),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(16),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(16),
        )

        self.bn7 = nn.BatchNorm2d(16)

    def forward(self, x):
        """
        bs, n_chs, seq_len = 128, 14, 32
        x = torch.rand(bs, 1, n_chs, seq_len).float()
        model = SegmentConv2D()
        y = model(x)  # (-1, 16)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Gloabl average
        x = self.bn7(x.mean(dim=(-2, -1), keepdims=True)).squeeze()

        return x  # (-1, 16)


if __name__ == "__main__":
    ## Demo data
    bs, n_chs, seq_len = 128, 14, 32
    x = torch.rand(bs, 1, n_chs, seq_len).float()
    model = SegmentConv2D()
    y = model(x)  # (-1, 16)
    print(y.shape)

    print(summary(model, x.shape))

    print(y.mean(), y.std())
    # In [16]:     print(y.mean(), y.std())
    # tensor(4.3365e-08, grad_fn=<MeanBackward0>) tensor(1.0004, grad_fn=<StdBackward0>)
