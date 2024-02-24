#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-24 12:04:38 (ywatanabe)"

import math

import torch
import torch.nn as nn


class Transformer(nn.Module):
    """
    A Transformer model for sequence-to-sequence learning.

    Attributes:
        model_config (dict): Configuration for the transformer model.
        encoder_embedding (nn.Linear): Embedding layer for the encoder.
        pos_encoder (PositionalEncoding): Positional encoding layer.
        transformer_encoder (nn.TransformerEncoder): Transformer encoder.
        transformer_decoder (nn.TransformerDecoder): Transformer decoder.
        fc_out (nn.Linear): Final fully connected output layer.
        activation (dict): Dictionary to store activations from hooks.

    Examples:
        BATCH_SIZE = 128
        input_dim = 16
        seq_len = 7

        x = torch.rand(BATCH_SIZE, input_dim, seq_len)

        model_config = {
            "tf_input_dim": input_dim,
            "tf_hidden_dim": 64,
            "tf_n_heads": 2,
            "tf_dim_feedforward": 2048,
            "tf_n_enc_layers": 2,
            "tf_n_dec_layers": 2
        }

        m = TransformerModel(model_config)
        y = m(x)  # Shape: [BATCH_SIZE, 1, input_dim]
    """

    def __init__(
        self,
        model_config,
    ):
        super(Transformer, self).__init__()
        self.model_config = model_config

        # Encoder
        self.encoder_embedding = nn.Linear(
            self.model_config["tf_input_dim"],
            self.model_config["tf_hidden_dim"],
        )
        self.pos_encoder = PositionalEncoding(
            d_model=self.model_config["tf_hidden_dim"]
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_config["tf_hidden_dim"],
            nhead=self.model_config["tf_n_heads"],
            dim_feedforward=self.model_config["tf_dim_feedforward"],
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.model_config["tf_n_enc_layers"]
        )

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.model_config["tf_hidden_dim"],
            nhead=self.model_config["tf_n_heads"],
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=self.model_config["tf_n_dec_layers"]
        )

        # FC
        self.fc_out = nn.Linear(
            self.model_config["tf_hidden_dim"],
            self.model_config["tf_input_dim"],
        )

        # To trap_activation latent features
        self.activation = {}

    @staticmethod
    def reshape(data):
        # [batch_size, seq_len n_features] to [seq_len, batch_size, n_features]
        return data.permute(2, 0, 1)

    def forward(self, pre, get_latent=False):
        return self._get_latent(pre) if get_latent else self._forward(pre)

    def _forward(self, pre):
        """pre: [batch_size, n_features, seq_len ]"""

        # Encoder
        pre = self.reshape(pre)
        pre = self.encoder_embedding(pre)
        pre = self.pos_encoder(pre)
        memory = self.transformer_encoder(pre)

        # Decoder
        tgt = pre
        out = self.transformer_decoder(tgt, memory)
        out = self.fc_out(out)
        return out[-1]

    def _get_latent(self, pre):
        self.transformer_encoder.layers[1].norm2.register_forward_hook(
            self.trap_activation("encoder_norm")
        )
        _ = self._forward(pre)
        return self.activation["encoder_norm"].transpose(0, 1).transpose(1, 2)
        # (-1, 64, 7) = (-1, 448)

    def trap_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()

        return hook


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


if __name__ == "__main__":
    from scripts.ml.parse_model_configs import parse_model_configs

    # Demo signal
    BATCH_SIZE = 128
    input_dim = 16
    seq_len = 7

    x = torch.rand(BATCH_SIZE, input_dim, seq_len)

    # Forwarding
    model_config = parse_model_configs()
    m = Transformer(
        model_config,
    )
    y = m(x)  # [-1, 1, 16]
    print(y.shape)

    l = m(x, get_latent=True)
    # [-1, 64, 7] # [-1, tf_hidden_dim, n_input_steps]
    print(l.shape)
