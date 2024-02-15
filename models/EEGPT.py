#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-16 06:46:17 (ywatanabe)"

import warnings

import mngs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.commitment_loss import commitment_loss
from models.LossBalancer import LossBalancer
from models.MultiTaskLoss import MultiTaskLoss

# from models.MNet import MNet
from models.PositionalEncodingLayer import PositionalEncodingLayer
from models.SplitConv import SplitConv
from models.Transformer import Transformer
from models.VectorQuantizer import VectorQuantizer
from sklearn.metrics import balanced_accuracy_score
from torchinfo import summary


class EEGPT(nn.Module):
    """
    EEGPT model for EEG signal processing using a GPT-like architecture.
    """

    def __init__(self, model_config={}):  # fixme
        super().__init__()
        model_config = dict(
            num_embeddings=4,
            embedding_dim=16,
            window_size=32,
        )
        self.model_config = model_config
        CONFIG = mngs.io.load_configs(show=False)

        # Convolutional Layer
        self.split_conv = SplitConv(self.model_config)

        # Vector Quantizer
        self.VQ = VectorQuantizer(
            num_embeddings=self.model_config["num_embeddings"],
            embedding_dim=self.model_config["embedding_dim"],
        )

        # Positional Encoding
        n_steps = int(
            CONFIG["WINDOW_SIZE_SEC"]
            * CONFIG["TGT_FS"]
            / self.model_config["window_size"]
        )

        # Transformer layers
        self.transformer = Transformer(
            input_dim=self.model_config["embedding_dim"],
            hidden_dim=64,
            nheads=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
            output_dim=28,
        )

        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
        self.loss_balancer = LossBalancer(self.model_config["num_embeddings"])
        self.multi_task_loss = MultiTaskLoss(
            [True, False, False]
        )  # not working well

        # To trap latent features
        self.activation = {}

    def forward(self, x, get_latent=False, alpha=1, beta=1, gamma=1):
        """x.shape # [-1, 14, 256]"""
        if self.training:
            self.multi_task_loss.train()
        else:
            self.multi_task_loss.eval()

        # Convolution
        xs = self.split_conv(x)  # len(xs) # 8; xs[0].shape # [32, 64]
        import ipdb

        ipdb.set_trace()
        # Quantization
        _q, commitment_losses, entropy_losses = zip(*[self.VQ(x) for x in xs])
        _q = torch.stack(_q, axis=1)
        # (-1, 8, self.model_config["embedding_dim"])
        commitment_loss = torch.mean(torch.stack(commitment_losses)).to(
            x.device
        )
        entropy_loss = torch.mean(torch.stack(entropy_losses)).to(x.device)

        # Split into pre and post
        q_pre = _q[:, :-1, :]
        q_post = _q[:, -1, :].unsqueeze(1)

        # For transfer learning
        if not get_latent:
            y = self.transformer(q_pre)
            # (-1, 1, self.model_config["embedding_dim"])

            prediction_loss = F.mse_loss(y, q_post)

            return (
                alpha * commitment_loss,
                beta * entropy_loss,
                gamma * prediction_loss,
            )

            # softmax_probs = F.softmax(logits, dim=-1).squeeze()
            # predicted_classes = softmax_probs.argmax(dim=-1)
            # true_classes = x2.squeeze().argmax(dim=-1)

            # # Classification loss
            # loss_clf = self.loss_balancer(
            #     self.cross_entropy_loss(logits.squeeze(), x2.squeeze()),
            #     true_classes,
            #     train=self.training,
            # )

            # # Diversity loss
            # avg_probs = softmax_probs.mean(dim=0)
            # loss_diversity = -torch.sum(
            #     avg_probs * torch.log(avg_probs + 1e-10)
            # )

            # # Commitment loss
            # loss_commitment = commitment_loss(x2)

            # Total loss
            # multi_task_loss = self.multi_task_loss(
            #     loss_clf, loss_diversity, loss_commitment
            # )  # it seems not working
            # multi_task_loss = self.multi_task_loss(loss_clf, loss_commitment)
            # multi_task_loss = loss_clf

            # # Calculate balanced accuracy for the last-sequency prediction
            # with warnings.catch_warnings():
            #     warnings.simplefilter("ignore")
            #     bacc = balanced_accuracy_score(
            #         true_classes.cpu().detach().numpy(),
            #         predicted_classes.cpu().detach().numpy(),
            #     )
            # return multi_task_loss, bacc

        else:
            self.transformer.transformer_encoder.layers[
                1
            ].norm2.register_forward_hook(self.get_latent("encoder_norm"))
            logits = self.transformer(q_pre)
            l1 = (
                self.activation["encoder_norm"].transpose(0, 1).transpose(1, 2)
            )
            return l1  # (-1, 64, 7) = (-1, 448)

    def get_latent(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()

        return hook


if __name__ == "__main__":
    m = EEGPT().cuda()
    x = torch.randn(32, 14, 256).cuda()
    loss = m(x)
    # Pretrained model
    # weights_path = (
    #     "./scripts/ml/pretrain_EEGPT/2024-02-11-07-05_dPXGGb4b/
    # )
    # m.load_state_dict(mngs.io.load(weights_path).state_dict())
    l = m(x, get_latent=True)
