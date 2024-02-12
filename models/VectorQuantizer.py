#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-12 23:19:45 (ywatanabe)"

import math

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
        # self.embedding.weight.data.uniform_(-1, 1)
        stdv = 1.0 / math.sqrt(self.embedding_dim)
        self.embedding.weight.data.normal_(mean=0, std=stdv)

    def forward(self, x):
        # self.embedding.weight.data = F.normalize(
        #     self.embedding.weight, p=2, dim=1
        # )
        # import ipdb; ipdb.set_trace()
        distances = self.calc_distances(x, self.embedding.weight)
        # self.softmax_distances = F.softmax(-distances, dim=1)
        self.q = F.gumbel_softmax(-distances, dim=1, hard=False)
        # self.q = F.gumbel_softmax(
        #     distances, dim=1, hard=False
        # )  # '-' distances would be correct
        return self.q
        # x_quantized = torch.matmul(
        #     self.softmax_distances, self.embedding.weight
        # )
        # return x_quantized

    # def forward(self, x):
    #     self.embedding.weight.data = F.normalize(
    #         self.embedding.weight, p=2, dim=1
    #     )
    #     cos = self.calc_cosine_similarity(x, self.embedding.weight)
    #     # self.softmax_distances = F.softmax(-distances, dim=1)
    #     # self.q = F.gumbel_softmax(cos, dim=1, hard=False)
    #     self.q = F.softmax(cos, dim=1)
    #     # x_quantized = torch.matmul(
    #     #     self.softmax_distances, self.embedding.weight
    #     # )
    #     # return x_quantized
    #     return self.q

    @staticmethod
    def calc_distances(x, y):
        norm_x = torch.sum(x**2, dim=1, keepdim=True)
        norm_embedding = torch.sum(y**2, dim=1)
        dot_product = torch.matmul(x, y.t())
        distances = norm_x + norm_embedding - 2 * dot_product
        return distances

    @staticmethod
    def calc_cosine_similarity(x, y):
        # Normalize input vectors and embedding vectors to have unit norm
        x_norm = F.normalize(x, p=2, dim=1)
        y_norm = F.normalize(y, p=2, dim=1)

        # Calculate cosine similarity
        cosine_sim = torch.matmul(x_norm, y_norm.t())

        # Transform similarities to 'distances' for softmax
        distances = 1 - cosine_sim
        return distances

    # @staticmethod
    # def calc_cosine_similarity(x, y):
    #     # Normalize input vectors and embedding vectors to have unit norm
    #     x_norm = F.normalize(x, p=2, dim=1)
    #     y_norm = F.normalize(y, p=2, dim=1)

    #     # Calculate cosine similarity
    #     cosine_sim = torch.matmul(x_norm, y_norm.t())

    #     # Since the softmax function expects 'distances', we can invert the cosine similarities
    #     # so that higher similarities (closer to 1) are treated as smaller 'distances'.
    #     return cosine_sim


if __name__ == "__main__":
    # Demo input vector data
    BATCH_SIZE, VEC_LEN = 16, 4096
    x = torch.randn(BATCH_SIZE, VEC_LEN).cuda()
    NUM_EMBEDDINGS = (
        64  # Each vector will be represented in one of these patterns.
    )

    m = VectorQuantizer(NUM_EMBEDDINGS, VEC_LEN).cuda()
    out = m(x)  # (BATCH_SIZE, VEC_LEN)

    out.sum().backward()

    print(m.embedding.weight)
    print(m.softmax_distances)
