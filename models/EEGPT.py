#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-01-25 11:00:20 (ywatanabe)"

""""To tokenize EEG signals and process them in a manner similar to GPT (Generative Pre-trained Transformer) used in Natural Language Processing (NLP)"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x


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
        x = torch.randn(BATCH_SIZE, VEC_LEN)
        NUM_EMBEDDINGS = 64  # Each vector will be represented in one of these patterns.

        m = VectorQuantizer(NUM_EMBEDDINGS, VEC_LEN)
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


class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, num_classes):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, src):
        src = src * torch.sqrt(torch.tensor(src.size(-1), dtype=torch.float32))
        output = self.transformer_encoder(src)
        output = self.fc(output.mean(dim=1))
        return output


class EEGEndToEndModel(nn.Module):
    def __init__(self):
        super(EEGEndToEndModel, self).__init__()
        self.feature_extractor = CNNFeatureExtractor()
        self.quantizer = VectorQuantizer(num_embeddings=16, embedding_dim=8)
        self.transformer = TransformerModel(
            input_dim=256, num_heads=4, num_layers=2, num_classes=10
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        print(f"features: {features.shape}")
        quantized = self.quantizer(features)
        print(f"quantized: {quantized.shape}")
        import ipdb

        ipdb.set_trace()
        output = self.transformer(quantized)
        print(f"output: {output.shape}")
        return output


if __name__ == "__main__":

    # Dummy input tensor (batch_size, vec_len)
    x = torch.randn(16, 4096)  # Example input
    # o = m(x)

    quantized_x = vector_quantizer(x)
    print(summary(vector_quantizer, [x.shape]))
    torch.unique(quantized_x)

    # model = EEGEndToEndModel()
    # eeg_data = torch.randn(64, 14, 256)  # Example EEG data
    # output = model(eeg_data)

    # vq = VectorQuantize(
    #     dim=256,
    #     codebook_size=512,  # codebook size
    #     decay=0.8,  # the exponential moving average decay, lower means the dictionary will change faster
    #     commitment_weight=1.0,  # the weight on the commitment loss
    # )
