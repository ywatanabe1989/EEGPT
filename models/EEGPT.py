#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-24 12:07:54 (ywatanabe)"

import warnings

import mngs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from models.LossBalancer import LossBalancer
from models.SplitConv import SplitConv
from models.Transformer import Transformer
from models.VectorQuantizer import VectorQuantizer


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

        # Convolutional Layer
        self.split_conv = SplitConv(self.model_config)

        # Vector Quantizer
        self.vq = VectorQuantizer(self.model_config)

        # Transformer layers
        self.transformer = Transformer(self.model_config)

        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")

        # Load weights
        conv_vq_weights_path = model_config.get("conv_vq_weights_path")
        weights_path = model_config.get("weights_path")

        # Check if both paths are provided and not None
        if conv_vq_weights_path is not None and weights_path is not None:
            raise ValueError(
                "Both 'conv_vq_weights_path' and 'weights_path' are provided. Please specify only one."
            )

        # Proceed with the rest of the code
        if conv_vq_weights_path is not None:
            self.load_weights(conv_vq_weights_path)
            self.freeze_conv_layers()
            self.freeze_vq_layers()
            _ = mngs.ml.utils.check_params(self, show=True)

        if weights_path is not None:
            self.load_weights(weights_path)
            _ = mngs.ml.utils.check_params(self, show=True)

    def forward(self, x, get_latent=False):
        """x.shape # [-1, 14, 256]"""

        # Normalization as the first layers
        x = mngs.gen.to_z(x, axis=(1, 2))

        # Convolution
        x = self.split_conv(x)
        # [-1, 64, x.shape[-1] // model_config["conv_split_size_pts"]]

        # Vector Quantization
        (
            q_pre,
            q_post,
            commitment_loss,
            entropy_loss,
            diversity_loss,
        ) = self.apply_vq_for_split_segments(x)
        # [-1, 64, x.shape[-1] // model_config["conv_split_size_pts"]]
        # q_pre: [-1, model_config["vq_num_embeddings"], x.shape[-1] // model_config["conv_split_size_pts"] - 1]
        # q_post: [-1, model_config["vq_num_embeddings"], 1]

        # Transformer
        if get_latent:
            return self.transformer(q_pre, get_latent=True)

        else:
            y = self.transformer(q_pre)
            prediction_loss = F.mse_loss(y, q_post.squeeze())
            return (
                self.model_config["vq_prediction_coef"] * prediction_loss,
                self.model_config["vq_commitment_coef"] * commitment_loss,
                self.model_config["vq_entropy_coef"] * entropy_loss,
                self.model_config["vq_diversity_coef"] * diversity_loss,
            )

    def apply_vq_for_split_segments(self, x):
        _qs = []
        _commitment_losses = []
        _entropy_losses = []
        _diversity_losses = []
        _assignment_probs = []
        for i_seg in range(x.shape[-1]):
            _x = x[..., i_seg]
            _q, _commitment_loss, _entropy_loss, _diversity_loss = self.vq(_x)

            _qs.append(_q)
            _commitment_losses.append(_commitment_loss)
            _entropy_losses.append(_entropy_loss)
            _diversity_losses.append(_diversity_loss)
            _assignment_probs.append(self.vq.assignment_probs)

        # [-1, 8, self.model_config["num_embeddings"]]
        q = torch.stack(_qs, axis=1).permute(0, 2, 1)
        q_pre, q_post = q[..., :-1], q[..., -1].unsqueeze(-1)

        self.assignment_probs = torch.stack(_assignment_probs, axis=1)
        commitment_loss = torch.mean(torch.stack(_commitment_losses))
        entropy_loss = torch.mean(torch.stack(_entropy_losses))
        diversity_loss = torch.mean(torch.stack(_diversity_losses))

        return q_pre, q_post, commitment_loss, entropy_loss, diversity_loss

    def load_weights(self, weights_path):
        state_dict = mngs.io.load(weights_path)
        self.load_state_dict(mngs.io.load(weights_path).state_dict())
        print(f"\nWeights were successfully loaded: {weights_path}")

    def freeze_conv_layers(
        self,
    ):
        for param in list(self.split_conv.parameters()):
            param.requires_grad = False

    def freeze_vq_layers(
        self,
    ):
        for param in list(self.vq.parameters()):
            param.requires_grad = False


if __name__ == "__main__":
    from scripts.ml.parse_model_configs import parse_model_configs

    x = torch.randn(32, 14, 256).cuda()

    model_config = parse_model_configs()
    m = EEGPT(model_config).cuda()
    losses = m(x)

    # Pretrained model
    # weights_path = (
    #     "./scripts/ml/pretrain_EEGPT/2024-02-11-07-05_dPXGGb4b/
    # )
    # m.load_state_dict(mngs.io.load(weights_path).state_dict())
    l = m(x, get_latent=True)  # (-1, 64, 7)
