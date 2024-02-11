#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-12 07:39:16 (ywatanabe)"

import torch.nn as nn
from torch.nn import Transformer


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        nheads,
        num_encoder_layers,
        num_decoder_layers,
    ):
        super(TransformerModel, self).__init__()
        self.transformer = Transformer(
            d_model=input_dim,
            nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
        )
        self.fc_out = nn.Linear(input_dim, input_dim)

    def forward(self, src):
        # src: (sequence length, batch size, input dimension)
        src = src.permute(
            1, 0, 2
        )  # Transformer expects src to be of shape (S, N, E)
        output = self.transformer(src, src)
        output = self.fc_out(output)
        return output[-1, :, :].unsqueeze(1)
