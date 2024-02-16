#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-16 08:04:13 (ywatanabe)"

import warnings

import mngs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.commitment_loss import commitment_loss
from models.LossBalancer import LossBalancer

# from models.MNet import MNet
from models.PositionalEncodingLayer import PositionalEncodingLayer
from models.SplitConv import SplitConv
from models.Transformer import Transformer
from models.VectorQuantizer import VectorQuantizer
from sklearn.metrics import balanced_accuracy_score
from torchinfo import summary

# from models.MultiTaskLoss import MultiTaskLoss


class EEGPT(nn.Module):
    """
    EEGPT model for EEG signal processing using a GPT-like architecture.
    """

    def __init__(self, model_config):
        super().__init__()
        model_config.update(
            dict(
                window_size=32,
            )
        )
        self.model_config = model_config
        CONFIG = mngs.io.load_configs(show=False)

        # Convolutional Layer
        self.split_conv = SplitConv(self.model_config)

        # Vector Quantizer
        self.VQ = VectorQuantizer(
            num_embeddings=self.model_config["VQ_num_embeddings"],
            embedding_dim=self.model_config["VQ_embedding_dim"],
            softmax_type=self.model_config["VQ_softmax_type"],
            initializer=self.model_config["VQ_initializer"],
            weight_norm=self.model_config["VQ_weight_norm"],
        )

        # Positional Encoding
        n_steps = int(
            CONFIG["WINDOW_SIZE_SEC"]
            * CONFIG["TGT_FS"]
            / self.model_config["window_size"]
        )

        # Transformer layers
        self.transformer = Transformer(
            input_dim=self.model_config["VQ_embedding_dim"],
            hidden_dim=64,
            nheads=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
            output_dim=28,
        )

        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
        # self.loss_balancer = LossBalancer(self.model_config["num_embeddings"])
        # self.multi_task_loss = MultiTaskLoss(
        #     [True, False, False]
        # )  # not working well

        # To trap latent features
        self.activation = {}

    def forward(self, x, get_latent=False):
        """x.shape # [-1, 14, 256]"""
        # if self.training:
        #     self.multi_task_loss.train()
        # else:
        #     self.multi_task_loss.eval()

        # Convolution
        xs = self.split_conv(x)  # len(xs) # 8; xs[0].shape # [32, 64]

        # Quantization
        _q, _commitment_losses, _entropy_losses = zip(
            *[self.VQ(x) for x in xs]
        )
        _q = torch.stack(_q, axis=1)
        # (-1, 8, self.model_config["embedding_dim"])
        q_pre, q_post = _q[:, :-1, :], _q[:, -1, :].unsqueeze(1)

        # For transfer learning
        if not get_latent:
            y = self.transformer(q_pre)
            # (-1, 1, self.model_config["embedding_dim"])
            commitment_loss = torch.mean(torch.stack(_commitment_losses)).to(
                x.device
            )
            entropy_loss = torch.mean(torch.stack(_entropy_losses)).to(
                x.device
            )
            prediction_loss = F.mse_loss(y, q_post)
            return (
                self.model_config["loss_alpha"] * commitment_loss,
                self.model_config["loss_beta"] * entropy_loss,
                self.model_config["loss_gamma"] * prediction_loss,
            )

            # # Diversity loss
            # avg_probs = softmax_probs.mean(dim=0)
            # loss_diversity = -torch.sum(
            #     avg_probs * torch.log(avg_probs + 1e-10)
            # )

        else:
            self.transformer.transformer_encoder.layers[
                1
            ].norm2.register_forward_hook(self.get_latent("encoder_norm"))
            _ = self.transformer(q_pre)
            return (
                self.activation["encoder_norm"].transpose(0, 1).transpose(1, 2)
            )
            # (-1, 64, 7) = (-1, 448)

    def get_latent(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()

        return hook


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train the EEGPT model with given hyperparameters."
    )
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=1.0,
        help="Weight for the commitment loss",
    )
    parser.add_argument(
        "-b",
        "--beta",
        type=float,
        default=1.0,
        help="Weight for the entropy loss",
    )
    parser.add_argument(
        "-g",
        "--gamma",
        type=float,
        default=1.0,
        help="Weight for the prediction loss",
    )
    parser.add_argument(
        "--softmax_type",
        default="softmax",
        choices=["softmax", "gumbel_softmax_soft", "gumbel_softmax_hard"],
        help="Type of softmax to be used in the VQ layer: 'softmax', 'gumbel_softmax_soft', or 'gumbel_softmax_hard'.",
    )
    parser.add_argument(
        "--num_embeddings",
        default=4,
        help="",
    )
    parser.add_argument(
        "--embedding_dim",
        default=16,
        help="",
    )
    parser.add_argument(
        "--weight_norm",
        default=False,
        help="",
    )
    parser.add_argument(
        "--initializer",
        default="uniform",
        help="",
    )

    args = parser.parse_args()

    print(f"\nargs:\n{args}")

    model_config = dict(
        VQ_num_embeddings=args.num_embeddings,
        VQ_embedding_dim=args.embedding_dim,
        VQ_softmax_type=args.softmax_type,
        VQ_weight_norm=args.weight_norm,
        VQ_initializer=args.initializer,
        loss_alpha=args.alpha,
        loss_beta=args.beta,
        loss_gamma=args.gamma,
    )

    m = EEGPT(model_config).cuda()
    x = torch.randn(32, 14, 256).cuda()
    loss = m(x)
    # Pretrained model
    # weights_path = (
    #     "./scripts/ml/pretrain_EEGPT/2024-02-11-07-05_dPXGGb4b/
    # )
    # m.load_state_dict(mngs.io.load(weights_path).state_dict())
    l = m(x, get_latent=True)
