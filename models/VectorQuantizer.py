#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-23 21:33:59 (ywatanabe)"

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    A flexible Vector Quantizer module that converts input vectors into a combination of a finite number of embeddings,
    which are learnable parameters of the model. It supports different initialization methods for embeddings,
    application of weight normalization, and choice of softmax type for soft quantization.

    The module normalizes input vectors to have mean 0 and standard deviation 1 before computing distances to embeddings.
    It supports standard softmax, soft, and hard Gumbel softmax for computing soft assignment probabilities.

    Parameters:
        num_embeddings (int): The number of distinct embeddings.
        embedding_dim (int): The dimensionality of each embedding vector.
        softmax_type (str): Type of softmax for computing assignment probabilities. Supported types are 'softmax',
            'gumbel_softmax_soft', and 'gumbel_softmax_hard'.
        initializer (str): Method to initialize embeddings. Supported methods are 'uniform' and 'normal'.
        apply_weight_norm (bool): If True, applies weight normalization to embeddings before quantization.

    Attributes:
        num_embeddings (int): The number of embeddings.
        embedding_dim (int): The dimensionality of embeddings.
        embedding (nn.Embedding): The embedding layer.
        softmax_type (str): The chosen softmax type.
        initializer (str): The chosen initializer method.
        apply_weight_norm (bool): Flag to apply weight normalization.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: A tuple containing the quantized output tensor with shape (batch_size, embedding_dim),
        the commitment loss, and the entropy loss.

    Example usage:
        BATCH_SIZE, VEC_LEN = 16, 4096
        x = torch.randn(BATCH_SIZE, VEC_LEN).cuda()
        NUM_EMBEDDINGS = 64

        vector_quantizer = VectorQuantizer(NUM_EMBEDDINGS, VEC_LEN, softmax_type="softmax", initializer="uniform", apply_weight_norm=False).cuda()
        quantized_outputs, commitment_loss, entropy_loss = vector_quantizer(x)

        # Backpropagate
        commitment_loss.backward()

    Note:
        This implementation is designed for PyTorch and is suitable for gradient-based optimization. It provides a
        modular and flexible approach to vector quantization, allowing for experimentation with different quantization strategies.
    """

    def __init__(
        self,
        model_config,
    ):
        super(VectorQuantizer, self).__init__()

        self.model_config = model_config

        self.embedding = nn.Embedding(
            self.model_config["vq_num_embeddings"],
            self.model_config["vq_embedding_dim"],
        )

        if self.model_config["vq_initializer"] == "uniform":
            self.embedding.weight.data.uniform_(-1, 1)
        else:
            stdv = 1.0 / math.sqrt(self.model_config["vq_embedding_dim"])
            self.embedding.weight.data.normal_(mean=0, std=stdv)

    def forward(self, x):
        z = (x - x.mean(dim=-1, keepdims=True)) / (
            x.std(dim=-1, keepdims=True) + 1e-8
        )

        # Weight normalization
        if self.model_config["vq_weight_norm"]:
            self.embedding.weight.data = F.normalize(
                self.embedding.weight, p=2, dim=1
            )

        # Calc allocation based on distances
        distances = self.calc_distances(z, self.embedding.weight)
        self.assignment_probs = self.softmax(
            distances, self.model_config["vq_softmax_type"]
        )

        # Quantization based on allocation
        q = torch.matmul(self.assignment_probs, self.embedding.weight)

        # Guide the model to have steep probabilities to a reference point
        self.entropy_loss = self.calc_entropy_loss(self.assignment_probs)

        # Guide the model to make the quantization to reproduce the input
        self.commitment_loss = F.mse_loss(z, q)

        # Guide the model to make the quantization to reproduce the input
        self.diversity_loss = self.calc_diversity_loss(
            self.embedding.weight.data
        )

        return q, self.commitment_loss, self.entropy_loss, self.diversity_loss

    @staticmethod
    def softmax(distances, softmax_type):
        if softmax_type == "softmax":
            return F.softmax(-distances, dim=1)

        elif softmax_type == "gumbel_softmax_soft":
            return F.gumbel_softmax(-distances, dim=1, hard=False)

        elif softmax_type == "gumbel_softmax_hard":
            return F.gumbel_softmax(-distances, dim=1, hard=True)

        else:
            raise ValueError(f"Unknown softmax_type: {softmax_type}")

    @staticmethod
    def calc_entropy_loss(softmax):
        log_q = torch.log(softmax + 1e-10)
        entropy = -torch.sum(softmax * log_q, dim=1)
        return -entropy.mean()

    @staticmethod
    def calc_distances(x, y):
        norm_x = torch.sum(x**2, dim=1, keepdim=True)
        norm_embedding = torch.sum(y**2, dim=1)
        dot_product = torch.matmul(x, y.t())
        distances = norm_x + norm_embedding - 2 * dot_product
        return distances

    @staticmethod
    def calc_diversity_loss(embeddings, epsilon=1e-8):
        dists = torch.cdist(embeddings, embeddings, p=2) + epsilon
        inv_dists = 1 / dists
        mask = torch.eye(embeddings.size(0)).to(embeddings.device)
        masked_inv_dists = inv_dists.masked_fill_(mask == 1, 0)
        diversity_loss = masked_inv_dists.mean()
        return diversity_loss


if __name__ == "__main__":
    from scripts.ml.parse_model_configs import parse_model_configs

    model_config = parse_model_configs()

    # model_config = args_dict

    # Demo input vector data
    BATCH_SIZE, VEC_LEN = 32, 16
    x = torch.randn(BATCH_SIZE, VEC_LEN).cuda()

    # Feeds the demo data
    m = VectorQuantizer(model_config).cuda()

    q, commitment_loss, entropy_loss, diversity_loss = m(x)

    loss = commitment_loss.mean() + entropy_loss.mean()
    loss.backward()
