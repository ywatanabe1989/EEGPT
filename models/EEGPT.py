#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-17 22:54:19 (ywatanabe)"

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

        device = x.device

        # Normalization after cropping
        x = mngs.gen.to_z(x, axis=(1, 2))

        # Convolution
        xs = self.split_conv(x)  # len(xs) # 8; xs[0].shape # [32, 64]

        # Quantization
        qs = []
        _commitment_losses = []
        _entropy_losses = []
        assignment_probs = []
        for _x in xs:
            _q, _commitment_loss, _entropy_loss = self.VQ(_x)
            qs.append(_q)
            _commitment_losses.append(_commitment_loss)
            _entropy_losses.append(_entropy_loss)
            assignment_probs.append(self.VQ.assignment_probs)

        # _q, _commitment_losses, _entropy_losses = zip(
        #     *[self.VQ(x) for x in xs]
        # )
        self.assignment_probs = torch.stack(
            assignment_probs, axis=1
        )  # [-1, 8, self.model_config["num_embeddings"]]
        q = torch.stack(qs, axis=1).permute(0, 2, 1)
        # (-1, 8, self.model_config["embedding_dim"])
        q_pre, q_post = q[..., :-1], q[..., -1].unsqueeze(1)

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
        for param in list(self.VQ.parameters()):
            param.requires_grad = False

    # def freeze_vq_layers(self,):
    #     for param in list(self.VQ.parameters()):
    #         param.requires_grad = False


if __name__ == "__main__":
    model_config = dict(
        VQ_num_embeddings=4,
        VQ_embedding_dim=16,
        VQ_softmax_type="gumbel_softmax_hard",
        VQ_weight_norm=False,
        VQ_initializer="uniform",
        loss_commit_coef=1.0,
        loss_entropy_coef=0.3,
        loss_pred_coef=0.0,
        conv_vq_weights_path="./scripts/ml/pretrain_EEGPT/2024-02-17-01-48_KTyjYpl1/checkpoint.pth",
        weights_path=None,
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
