#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-12 07:30:25 (ywatanabe)"

""""To tokenize EEG signals and process them in a manner similar to GPT (Generative Pre-trained Transformer) used in Natural Language Processing (NLP)"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from models.MNet import MNet
from models.PositionalEncodingLayer import PositionalEncodingLayer
from models.SplitConv import SplitConv
from models.TransformerModel import TransformerModel
from models.VectorQuantizer import VectorQuantizer
from torchinfo import summary


class EEGPT(nn.Module):
    def __init__(self, model_config={}):  # fixme
        super().__init__()
        model_config = dict(
            num_embeddings=2,
            embedding_dim=64,
            window_size=32,
        )
        CONFIG = mngs.io.load_configs(show=False)

        self.model_config = model_config

        # Convolutional Layer
        self.split_conv = SplitConv(self.model_config)

        # Vector Quantizer
        self.VQ = VectorQuantizer(
            num_embeddings=self.model_config["num_embeddings"],
            embedding_dim=self.model_config["embedding_dim"],
        )  # [-1, 8*64]

        # Positional Encoding
        n_steps = int(
            CONFIG["WINDOW_SIZE_SEC"]
            * CONFIG["TGT_FS"]
            / self.model_config["window_size"]
        )

        # self.PE = self.generate_positional_encodings(n_steps, input_dim)
        self.PE = PositionalEncodingLayer(
            n_steps, self.model_config["embedding_dim"]
        )

        # Transformer layers
        self.transformer = TransformerModel(
            input_dim=64,
            hidden_dim=32,
            nheads=1,
            num_encoder_layers=1,
            num_decoder_layers=1,
        )

        # To trap latent features
        self.activation = {}

    def forward(self, x, get_latent=False):
        # x.shape # [-1, 14, 256]
        # conv
        xs = self.split_conv(x)
        # len(xs) # 8; xs[0].shape # [32, 64]

        xs = [self.VQ(x) for x in xs]
        # len(xs) # 8; xs[0].shape # [32, 64]

        x = torch.stack(xs, axis=1)  # (32, 8, 64) # seq_len = 8

        # positional encoding
        x += self.PE()

        # Split x to two parts for self-supervised training
        x1 = x[:, :-1, :]  # (64, 7, 64)
        x2 = x[:, -1, :].unsqueeze(1)  # (64, 1, 64)

        # For transfer learning
        if not get_latent:
            y = self.transformer(x1)  # (7, 64, 64)
        else:
            self.transformer.transformer.encoder.norm.register_forward_hook(
                self.get_latent("encoder_norm")
            )
            y = self.transformer(x1)
            l1 = (
                self.activation["encoder_norm"].transpose(0, 1).transpose(1, 2)
            )
            return l1  # (-1, 64, 7) = (-1, 448)

        # From the first 7 segments, the last segment is predicted.
        loss = ((y - x2) ** 2).mean()

        return loss

    def get_latent(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()

        return hook


if __name__ == "__main__":
    m = EEGPT().cuda()
    x = torch.randn(32, 14, 256).cuda()
    loss = m(x)
    # for _ in range(10):
    #     x = torch.randn(32, 14, 256).cuda()
    #     loss = m(x)
    #     loss.backward()
    #     print(loss)

    # Pretrained model
    # weights_path = (
    #     "./scripts/ml/pretrain_EEGPT/2024-02-11-07-05_dPXGGb4b/
    # )
    # m.load_state_dict(mngs.io.load(weights_path).state_dict())
    l = m(x, get_latent=True)

    # m.conv_and_vq(x)
    print(summary(m, x.shape))
