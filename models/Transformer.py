#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-15 18:33:37 (ywatanabe)"

import math
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        # Pre-compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # Make the positional encoding as long as the input sequence
        x_pe = self.pe[:, : x.size(1)]
        return x + x_pe


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim=196,
        hidden_dim=64,
        nheads=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
        output_dim=28,
    ):
        super(Transformer, self).__init__()
        self.hidden_dim = hidden_dim

        # Encoder
        self.encoder_embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(d_model=hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nheads, dim_feedforward=2048
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=2)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=2
        )

        # FC
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    @staticmethod
    def reshape(data):
        # [batch_size, seq_len n_features] to [seq_len, batch_size, n_features]
        return data.permute(1, 0, 2)

    def forward(self, pre):
        """pre: [batch_size, seq_len n_features]"""

        # Encoder
        pre = self.reshape(pre)
        pre = self.encoder_embedding(pre)
        pre = self.pos_encoder(pre)
        memory = self.transformer_encoder(pre)

        # Decoder
        tgt = pre
        out = self.transformer_decoder(tgt, memory)
        out = self.fc_out(out)
        return out[-1].unsqueeze(1)
