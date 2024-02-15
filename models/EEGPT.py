#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-15 09:11:34 (ywatanabe)"

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
from models.TransformerModel import TransformerModel
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

        # self.PE = self.generate_positional_encodings(n_steps, input_dim)
        self.PE = PositionalEncodingLayer(
            n_steps - 1, self.model_config["embedding_dim"]
        )  # self.model_config["embedding_dim"]

        # Transformer layers
        self.transformer = TransformerModel(
            input_dim=self.model_config["embedding_dim"],
            hidden_dim=32,
            nheads=4,
            num_encoder_layers=6,
            num_decoder_layers=6,
        )

        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
        self.loss_balancer = LossBalancer(self.model_config["num_embeddings"])
        self.multi_task_loss = MultiTaskLoss(
            [True, False, False]
        )  # not working well

        # To trap latent features
        self.activation = {}

    def forward(self, x, get_latent=False, alpha=1, beta=1):
        """x.shape # [-1, 14, 256]"""
        if self.training:
            self.multi_task_loss.train()
        else:
            self.multi_task_loss.eval()

        # Convolution
        xs = self.split_conv(x)  # len(xs) # 8; xs[0].shape # [32, 64]

        # Quantization
        xs_quantized, commitment_losses, entropy_losses = zip(
            *[self.VQ(x) for x in xs]
        )
        commitment_loss = torch.mean(torch.stack(commitment_losses)).to(
            x.device
        )
        entropy_loss = torch.mean(torch.stack(entropy_losses)).to(x.device)

        # To tensor
        x_quantized = torch.stack(
            xs_quantized, axis=1
        )  # (-1, 8, self.model_config["embedding_dim"])
        # Split x to two parts for self-supervised training
        xq1 = x_quantized[
            :, :-1, :
        ]  # (-1, 7, self.model_config["embedding_dim"])
        xq2 = x_quantized[:, -1, :].unsqueeze(
            1
        )  # (-1, 1, self.model_config["embedding_dim"])

        # Positional Encoding
        xq1 += self.PE()

        # For transfer learning
        if not get_latent:
            transformer_output = self.transformer(
                xq1
            )  # (-1, 1, self.model_config["embedding_dim"])

            prediction_loss = F.mse_loss(transformer_output, xq2)

            return (
                commitment_loss,
                alpha * entropy_loss,
                beta * prediction_loss,
            )

            # total_loss = (
            #     commitment_loss + alpha + entropy_loss + beta * prediction_loss
            # )

            # return total_loss

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
            self.transformer.transformer.encoder.norm.register_forward_hook(
                self.get_latent("encoder_norm")
            )
            logits = self.transformer(xq1)
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
    # print(summary(m, x.shape))

    # print(m.VQ.embedding.weight)
    print(m.VQ.q)
    print(m.VQ.q.argmax(axis=-1))

    # loss.sum().backward()
