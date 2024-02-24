#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-24 11:27:30 (ywatanabe)"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class SplitConv(nn.Module):
    """
    Split signal into segments based on model_config["conv_split_size_pts"] and applies SegmentConv2D for each part of the segments in a shared manner

    Example:
        bs, n_chs, seq_len = 128, 14, 256
        x = torch.rand(bs, 1, n_chs, seq_len).float()
        m = SegmentConv2D({
            "conv_split_size_pts": 32,
            "conv_n_blocks": 3,
        })
        y = m(x)  # (-1, 16)
        print(y.shape)
    """

    def __init__(self, model_config):
        super(SplitConv, self).__init__()
        self.model_config = model_config
        self.SegmentConv2D = SegmentConv2D(model_config)

    def forward(self, x):
        window_size_pts = self.model_config["conv_split_size_pts"]
        xs = [
            self.SegmentConv2D(
                x[
                    ..., i * window_size_pts : (i + 1) * window_size_pts
                ].unsqueeze(1)
            )
            for i in range(x.shape[-1] // window_size_pts)
        ]
        return torch.stack(xs, dim=-1)


class SegmentConv2D(nn.Module):
    """
    14 channels x 32 time steps (= 250 ms at 128 Hz)-> 16 dimensional vector

    Example:
        bs, n_chs, seq_len = 128, 14, 32
        x = torch.rand(bs, 1, n_chs, seq_len).float()
        m = SegmentConv2D({"conv_n_blocks": 3})
        y = m(x)  # (-1, 16)
        print(y.shape)
    """

    @staticmethod
    def build_block(n_input):
        return nn.Sequential(
            nn.Conv2d(n_input, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(16),
        )

    def __init__(self, model_config):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i_block in range(model_config["conv_n_blocks"]):
            self.blocks.append(
                self.build_block(n_input=1 if i_block == 0 else 16)
            )

        self.bn_last = nn.BatchNorm2d(16)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)

        for block in self.blocks:
            x = block(x)

        x = self.bn_last(x.mean(dim=-1, keepdims=True))

        return x.squeeze()  # (-1, 16)


if __name__ == "__main__":
    from scripts.ml.parse_model_configs import parse_model_configs

    # Demo data
    bs, n_chs, seq_len = 128, 14, 256
    x = torch.rand(bs, n_chs, seq_len).float()

    # Forwarding
    model_config = parse_model_configs()
    model = SplitConv(model_config)
    y = model(x)  # (-1, 16, seq_len//model_config["conv_split_size_pts"])

    # Print results
    print(summary(model, x.shape))
    print(y.shape)
    print(y.mean(), y.std())
