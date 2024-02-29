#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-26 22:34:23 (ywatanabe)"

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
            dropout=self.model_config["tf_dropout_ratio"],
            # batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.model_config["tf_n_enc_layers"]
        )

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.model_config["tf_hidden_dim"],
            nhead=self.model_config["tf_n_heads"],
            dim_feedforward=self.model_config["tf_dim_feedforward"],
            dropout=self.model_config["tf_dropout_ratio"],
            # batch_first=True,
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

    # @staticmethod
    # def reshape(data):
    #     """[batch_size, n_features, seq_len] to [batch_size, seq_len, n_features]"""
    #     return data.transpose(1, 2)
    @staticmethod
    def reshape(data):
        """[batch_size, n_features, seq_len] to [seq_len, batch_size, n_features]"""
        return data.permute(2, 0, 1)  # [REVISED]

    def forward(self, pre, get_latent=False):
        if get_latent:
            return self._get_latent(pre)
        else:
            return self._forward(pre)

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
        out = out.permute(1, 2, 0)  # [REVISED]
        # out = out.transpose(1, 2)

        return out  # [..., -1]

    def _get_latent(self, pre):
        self.transformer_encoder.layers[1].norm2.register_forward_hook(
            self.trap_activation("latent")
        )
        _ = self._forward(pre)
        return self.activation["latent"].permute(1, 2, 0)  # [REVISED]
        # return self.activation["latent"].transpose(1, 2)

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
    model_config = mngs.io.load(
        "./scripts/ml/pretrain_EEGPT/FINISHED/2024Y-02M-25D-12h27m14s_9zs9/transformer_model_config.yaml"
    )
    model_config.pop("conv_vq_weights_path", None)
    weights_path = "./scripts/ml/pretrain_EEGPT/FINISHED/2024Y-02M-25D-12h27m14s_9zs9/transformer_checkpoint.pth"
    model_config["weights_path"] = weights_path

    # model_config = parse_model_configs()
    m = (
        Transformer(
            model_config,
        )
        .eval()
        .cuda()
    )
    # y = m(x)  # [-1, 16, seq_len]
    # print(y.shape)

    m(x.cuda())
    l = m(x.cuda(), get_latent=True)
    # [-1, 64, 7] # [-1, tf_hidden_dim, seq_len]
    print(l.shape)
