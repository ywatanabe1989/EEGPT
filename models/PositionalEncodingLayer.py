#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-12 07:28:14 (ywatanabe)"

import torch
import torch.nn as nn


class PositionalEncodingLayer(nn.Module):
    def __init__(self, length, d_model):
        super().__init__()
        self.length = length
        self.d_model = d_model
        PE = self.gen_PE()
        self.register_buffer("PE", PE)
        self.register_buffer("scale", torch.tensor([1]))
        # self.amplitude = torch.tensor([1])

    def forward(
        self,
    ):
        return self.PE.unsqueeze(0) * self.scale

    def gen_PE(self):
        PE = torch.zeros(self.length, self.d_model)
        position = torch.arange(0, self.length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float()
            * -(torch.log(torch.tensor(10000.0)) / self.d_model)
        )
        PE[:, 0::2] = torch.sin(position * div_term)
        PE[:, 1::2] = torch.cos(position * div_term)
        return PE


if __name__ == "__main__":
    seq_len = 8
    n_chs = 64
    m = PositionalEncodingLayer(8, 64)
    print(m())
    pe = m()
    import seaborn as sns

    sns.heatmap(pe.squeeze())
