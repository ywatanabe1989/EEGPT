#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-01-25 09:20:58 (ywatanabe)"

""""To tokenize EEG signals and process them in a manner similar to GPT (Generative Pre-trained Transformer) used in Natural Language Processing (NLP)"""


import torch
import torch.nn as nn
import torch.nn.functional as F

# from vector_quantize_pytorch import VectorQuantize


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
    A module that quantizes input vectors to a finite set of embeddings.

    # Example usage:
    num_embeddings = 4  # For 2-bit quantization
    embedding_dim = 64  # Dimensionality of the embedding space
    vector_quantizer = VectorQuantizer(num_embeddings, embedding_dim)
    x = torch.randn(16, 19, 1024)  # Example input
    quantized_x = vector_quantizer(x) # (16, 19, 1024)
    """

    def __init__(self, num_embeddings, embedding_dim):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)  # 4, 64
        self.embedding.weight.data.uniform_(-1, 1)  # (4, 64)

    def forward(self, x):
        # Flatten the input tensor to shape (batch_size * n_chs * seq_len, embedding_dim)
        batch_size, n_chs, seq_len = x.size()
        flat_x = x.view(-1, self.embedding_dim)  # (4864, 64)

        # Calculate distances between input and embedding weights
        distances = (
            torch.sum(flat_x**2, dim=1, keepdim=True)  # (4864, 1)
            + torch.sum(self.embedding.weight**2, dim=1)  # (4)
            - 2 * torch.matmul(flat_x, self.embedding.weight.t())  # (4864, 4)
        )

        # Find the closest embeddings
        closest_idxs = torch.argmin(distances, dim=1)

        # Quantize the input to the closest embeddings
        x_quantized = self.embedding(closest_idxs).view(
            batch_size, n_chs, seq_len
        )
        return x_quantized


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
    # Example usage:
    num_embeddings = 4  # For 2-bit quantization
    embedding_dim = 64  # Dimensionality of the embedding space
    vector_quantizer = VectorQuantizer(num_embeddings, embedding_dim)

    # Dummy input tensor (batch_size, n_chs, seq_len)
    x = torch.randn(16, 19, 1024)  # Example input
    quantized_x = vector_quantizer(x)

    # model = EEGEndToEndModel()
    # eeg_data = torch.randn(64, 14, 256)  # Example EEG data
    # output = model(eeg_data)

    # vq = VectorQuantize(
    #     dim=256,
    #     codebook_size=512,  # codebook size
    #     decay=0.8,  # the exponential moving average decay, lower means the dictionary will change faster
    #     commitment_weight=1.0,  # the weight on the commitment loss
    # )

    # x = torch.randn(1, 1024, 256)
    # quantized, indices, commit_loss = vq(x)  # (1, 1024, 256), (1, 1024), (1)
