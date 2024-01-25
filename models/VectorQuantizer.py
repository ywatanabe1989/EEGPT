#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-01-25 11:33:36 (ywatanabe)"

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Converts input vectors into combinations of finite number of embeddings,
    which are learnable parameters of the model.

    Arguments:
        num_embeddings (int): The number of distinct embeddings.
        embedding_dim (int): The dimensionality of each embedding vector.

    Attributes:
        num_embeddings (int): The number of embeddings.
        embedding_dim (int): The dimensionality of embeddings.
        embedding (nn.Embedding): The embedding layer containing the embeddings.

    Returns:
        Tensor: The quantized output tensor with shape (batch_size, embedding_dim).

    Example usage:
        # Demo input vector data
        BATCH_SIZE, VEC_LEN = 16, 4096
        x = torch.randn(BATCH_SIZE, VEC_LEN).cuda()
        NUM_EMBEDDINGS = 64  # Each vector will be represented in one of these patterns.

        m = VectorQuantizer(NUM_EMBEDDINGS, VEC_LEN).cuda()
        out = m(x)  # (BATCH_SIZE, VEC_LEN)

        # Differentiability check
        out.sum().backward()

    Note:
        This implementation is specific to PyTorch tensors and leverages soft quantization,
        making it suitable for gradient-based optimization.
    """

    def __init__(self, num_embeddings, embedding_dim):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1, 1)

    def forward(self, x):
        distances = self.calc_distances(x, self.embedding.weight)
        softmax_distances = F.softmax(-distances, dim=1)
        x_quantized = torch.matmul(softmax_distances, self.embedding.weight)
        return x_quantized

    @staticmethod
    def calc_distances(x, y):
        norm_x = torch.sum(x**2, dim=1, keepdim=True)
        norm_embedding = torch.sum(y**2, dim=1)
        dot_product = torch.matmul(x, y.t())
        distances = norm_x + norm_embedding - 2 * dot_product
        return distances


if __name__ == "__main__":
    # Demo input vector data
    BATCH_SIZE, VEC_LEN = 16, 4096
    x = torch.randn(BATCH_SIZE, VEC_LEN).cuda()
    NUM_EMBEDDINGS = (
        64  # Each vector will be represented in one of these patterns.
    )

    m = VectorQuantizer(NUM_EMBEDDINGS, VEC_LEN).cuda()
    out = m(x)  # (BATCH_SIZE, VEC_LEN)

    # Differentiability check
    out.sum().backward()
