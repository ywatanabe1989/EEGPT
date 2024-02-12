#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-12 23:43:51 (ywatanabe)"

import warnings

import mngs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.commitment_loss import commitment_loss
from models.LossWeighting import LossWeighting
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
            n_steps - 1, self.model_config["num_embeddings"]
        )  # self.model_config["embedding_dim"]

        # Transformer layers
        self.transformer = TransformerModel(
            input_dim=self.model_config["num_embeddings"],
            hidden_dim=32,
            nheads=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
        )

        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
        self.loss_weighting = LossWeighting(self.model_config["embedding_dim"])
        # self.multi_task_loss = MultiTaskLoss([True, False, False])
        self.multi_task_loss = MultiTaskLoss([True, False])

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
        xs = [self.VQ(x) for x in xs]  # len(xs) # 8; xs[0].shape # [32, 64]

        # To tensor
        x = torch.stack(
            xs, axis=1
        )  # (-1, 8, self.model_config["num_embeddings"])

        # Split x to two parts for self-supervised training
        x1 = x[:, :-1, :]  # (-1, 7, 4)
        x2 = x[:, -1, :].unsqueeze(1)  # (-1, 1, 4)

        # Positional Encoding
        x1 += self.PE()

        # For transfer learning
        if not get_latent:
            logits = self.transformer(
                x1
            )  # (-1, 1, self.model_config["num_embeddings"])
            softmax_probs = F.softmax(logits, dim=-1)
            # predicted_classes = torch.max(softmax_probs, dim=-1)[1].squeeze()
            predicted_classes = softmax_probs.squeeze().argmax(dim=-1)
            true_classes = x2.squeeze().argmax(dim=-1)

            # Classification loss
            loss_clf = self.cross_entropy_loss(
                logits.squeeze(), x2.squeeze()
            ).mean()
            # loss_clf = self.loss_weighting(
            #     loss_clf, true_classes, is_training=self.training
            # )

            # Diversity loss
            avg_probs = softmax_probs.mean(dim=0)
            loss_diversity = -torch.sum(
                avg_probs * torch.log(avg_probs + 1e-10)
            )

            # Commitment loss
            loss_commitment = commitment_loss(x2)

            # Total loss
            # multi_task_loss = self.multi_task_loss(
            #     loss_clf, loss_diversity, loss_commitment
            # )
            # multi_task_loss = self.multi_task_loss(loss_clf, loss_commitment)
            multi_task_loss = loss_clf

            # Calculate balanced accuracy for the last-sequency prediction
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bacc = balanced_accuracy_score(
                    true_classes.cpu().detach().numpy(),
                    predicted_classes.cpu().detach().numpy(),
                )

                # acc = np.mean(
                #     true_classes.cpu().detach().numpy()
                #     == predicted_classes.cpu().detach().numpy()
                # )

            return multi_task_loss, bacc
            # return multi_task_loss, acc

        else:
            self.transformer.transformer.encoder.norm.register_forward_hook(
                self.get_latent("encoder_norm")
            )
            logits = self.transformer(x1)
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
    print(m.VQ.softmax_distances)

    # loss.sum().backward()

    m.eval()
    loss = m(x)
