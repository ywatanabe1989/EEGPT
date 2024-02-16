#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-16 07:12:49 (ywatanabe)"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class SplitConv(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        self.Conv2DNet = Conv2DNet()  # [-1, 14, 32] -> [-1, 64]

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
        xs = [self.Conv2DNet(x.unsqueeze(1)) for x in xs]
        return xs


class Conv2DNet(nn.Module):
    """32 time steps of receptive field (= 250 ms at 128 Hz)"""

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)

        self.pool1 = nn.MaxPool2d(2, 2)
        # self.downsample1 = nn.Conv2d(16, 32, kernel_size=1)

        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(16)

        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(16)

        self.pool2 = nn.MaxPool2d(2, 2)
        # self.downsample2 = nn.Conv2d(16, 64, kernel_size=1)

        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(16)

        self.conv6 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(16)

        self.pool3 = nn.MaxPool2d(2, 2)
        # self.downsample3 = nn.Conv2d(64, 128, kernel_size=1)

        # self.conv7 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(16)

        # self.conv8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        # self.bn8 = nn.BatchNorm2d(128)

        # self.pool4 = nn.MaxPool2d(2, 2)
        # # self.fc = nn.Linear(128, 16)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # (16, 16, 14, 32)
        out = F.relu(self.bn2(self.conv2(out)))  # (16, 16, 14, 32)
        out = self.pool1(out)  # (16, 1, 7, 16)
        residual = self.pool1(x)  # (16, 16, 19, 64)
        out = out + residual  # (16, 16, 7, 32)

        out = self.dropout(out)

        residual = out
        out = F.relu(self.bn3(self.conv3(out)))  # (16, 32, 7, 16)
        out = F.relu(self.bn4(self.conv4(out)))  # (16, 32, 7, 16)
        out = self.pool2(out)  # (16, 32, 3, 8)
        residual = self.pool2(residual).mean(
            dim=1, keepdims=True
        )  # (16, 32, 3, 8)
        out = out + residual  # (16, 32, 3, 8)

        out = self.dropout(out)

        residual = out
        out = F.relu(self.bn5(self.conv5(out)))  # (16, 64, 3, 8)
        out = F.relu(self.bn6(self.conv6(out)))  # (16, 64, 3, 8)
        out = self.pool3(out)  # (16, 64, 1, 4)
        residual = self.pool3(residual).mean(
            dim=1, keepdims=True
        )  # (16, 64, 1, 4)
        out = out + residual  # (16, 64, 1, 4)

        out = self.dropout(out)
        # out = F.relu(self.bn6(self.conv6(out)))  # (16, 64, 3, 8)

        # residual = out  # (16, 64, 1, 4)
        # out = F.relu(self.bn7(self.conv7(out)))  # (16, 128, 1, 4)
        # out = F.relu(self.bn8(self.conv8(out)))  # (16, 128, 1, 4)
        # residual = residual.mean(dim=1, keepdims=True)  # (16, 1, 1, 4)
        # out = out + residual  # (16, 128, 1, 4)
        # out = out.mean(dim=-1, keepdims=True)
        # out = out.mean(dim=-1).squeeze().mean(dim=-1).squeeze().unsqueeze(-1)

        out = out.mean(dim=(-2, -1), keepdims=True)  # 16, 64
        out = self.bn7(out)

        out = self.dropout(out)

        return out.squeeze()


if __name__ == "__main__":
    ## Demo data
    bs, n_chs, seq_len = 128, 14, 32
    x = torch.rand(bs, 1, n_chs, seq_len).float()
    model = Conv2DNet()
    y = model(x)  # (16, 16)
    print(y.shape)

    print(summary(model, x.shape))

    print(y.mean(), y.std())
    # In [16]:     print(y.mean(), y.std())
    # tensor(4.3365e-08, grad_fn=<MeanBackward0>) tensor(1.0004, grad_fn=<StdBackward0>)
