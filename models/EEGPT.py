#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-17 10:13:32 (ywatanabe)"

import warnings

import mngs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from models.LossBalancer import LossBalancer
# from models.PositionalEncodingLayer import PositionalEncodingLayer
from models.SplitConv import SplitConv
from models.Transformer import Transformer
from models.VectorQuantizer import VectorQuantizer

# from sklearn.metrics import balanced_accuracy_score
# from torchinfo import summary


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

        # Transformer layers
        self.transformer = Transformer(
            input_dim=self.model_config["VQ_embedding_dim"],
            hidden_dim=64,
            nheads=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
        )

        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")

        # Load weights
        if model_config["conv_vq_weights_path"] is not None:
            self.load_conv_and_vq_and_freeze_them(
                model_config["conv_vq_weights_path"]
            )

    def forward(self, x, get_latent=False):
        """x.shape # [-1, 14, 256]"""

        device = x.device

        # Convolution
        xs = self.split_conv(x)  # len(xs) # 8; xs[0].shape # [32, 64]

        # Quantization
        _q, _commitment_losses, _entropy_losses = zip(
            *[self.VQ(x) for x in xs]
        )
        _q = torch.stack(_q, axis=1).permute(0, 2, 1)
        # (-1, 8, self.model_config["embedding_dim"])
        q_pre, q_post = _q[..., :-1], _q[..., -1].unsqueeze(1)

        # For transfer learning
        return (
            self._forward(
                q_pre, q_post, _commitment_losses, _entropy_losses, device
            )
            if not get_latent
            else self.transformer(q_pre, get_latent=True)
        )

    def _forward(
        self, q_pre, q_post, _commitment_losses, _entropy_losses, device
    ):
        y = self.transformer(q_pre)
        # (-1, self.model_config["embedding_dim"], 1)
        commitment_loss = torch.mean(torch.stack(_commitment_losses)).to(
            device
        )
        entropy_loss = torch.mean(torch.stack(_entropy_losses)).to(device)
        prediction_loss = F.mse_loss(y.squeeze(), q_post.squeeze()).to(device)

        # # Diversity loss
        # avg_probs = softmax_probs.mean(dim=0)
        # loss_diversity = -torch.sum(
        #     avg_probs * torch.log(avg_probs + 1e-10)
        # )

        return (
            self.model_config["loss_commit_coef"] * commitment_loss,
            self.model_config["loss_entropy_coef"] * entropy_loss,
            self.model_config["loss_pred_coef"] * prediction_loss,
        )

    def load_conv_and_vq_and_freeze_them(self, weights_path):
        self.load_state_dict(
            mngs.io.load(
                weights_path
                # # "./scripts/ml/pretrain_EEGPT/2024-02-16-06-07_ysWI49LM/checkpoint.pth"
                # "./scripts/ml/pretrain_EEGPT/2024-02-17-01-48_uDT2kC6q/checkpoint.pth"
            ).state_dict()
        )
        # Freeze loaded weights in the conv and vq layers
        for param in list(self.split_conv.parameters()) + list(
            self.VQ.parameters()
        ):
            param.requires_grad = False


if __name__ == "__main__":
    model_config = dict(
        VQ_num_embeddings=4,
        VQ_embedding_dim=16,
        VQ_softmax_type="gumbel_softmax_hard",
        VQ_weight_norm=False,
        VQ_initializer="uniform",
        loss_commit_coef=1.0,
        loss_entropy_coef=0.3,
        loss_pred_coef=0,
    )

    m = EEGPT(model_config).cuda()
    x = torch.randn(32, 14, 256).cuda()
    loss = m(x)
    # Pretrained model
    # weights_path = (
    #     "./scripts/ml/pretrain_EEGPT/2024-02-11-07-05_dPXGGb4b/
    # )
    # m.load_state_dict(mngs.io.load(weights_path).state_dict())
    l = m(x, get_latent=True)  # (-1, 64, 7)
