# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# Time-stamp: "2024-02-15 16:22:36 (ywatanabe)"

import math
import random

import matplotlib.pyplot as plt
import mngs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import umap
from models.VectorQuantizer import VectorQuantizer
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def split_MNIST(data):
    data = data.reshape(
        len(data), 4, -1
    )  # (1, 8, 98) # (batch_size, seq_len, n_features)
    # Provide only the first 21 rows as input
    pre = data[:, :-1, :]
    # The last 7 rows will be the target
    post = data[:, -1:, :]
    return pre, post


def load_MNIST_dataloaders():
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_data = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    # Split dataset into training and validation
    train_size = int(0.8 * len(mnist_data))
    val_size = len(mnist_data) - train_size
    train_dataset, val_dataset = random_split(
        mnist_data, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=1024, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1024, shuffle=False, drop_last=True
    )
    return train_loader, val_loader


def train_model(model, train_loader, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        labels = []
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.cuda().reshape(len(data), -1)
            optimizer.zero_grad()
            x_quantized, commitment_loss, entropy_loss = model(data)
            loss = commitment_loss + entropy_loss
            # loss = commitment_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.5f}")


def to_tensor(dl):
    all_data = []
    all_targets = []

    for i in range(len(dl.dataset)):
        data, target = dl.dataset[
            i
        ]  # Assuming each dataset item is a tuple (data, target)
        all_data.append(data)
        all_targets.append(torch.tensor(target))

    # Stack all tensors into a single large tensor
    all_data = torch.vstack(all_data)
    all_targets = torch.hstack(all_targets)

    assert len(all_data) == len(all_targets)
    return all_data, all_targets


if __name__ == "__main__":

    dl_tra, dl_val = load_MNIST_dataloaders()

    # Instantiate the VectorQuantizer
    num_embeddings = 10
    embedding_dim = dl_tra.dataset[0][0].reshape(-1).shape[0]
    model = VectorQuantizer(num_embeddings, embedding_dim).cuda()
    # Trainable: {'embedding.weight': [10, 784]}

    # Quantize the data
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, dl_tra, optimizer, num_epochs=10)

    data, targets = to_tensor(dl_val)
    data = data.reshape(len(data), -1)

    data_q, _, _ = model(data.cuda())
    data_q = data_q.detach().cpu().numpy()

    # data_q = model.q.detach().cpu().numpy()

    # Quantization
    UMAP = mngs.ml.clustering.UMAP()
    fig, embedding = UMAP.cluster(data_q, labels=targets, supervised=False)
    plt.show()

    # # Original
    # UMAP = mngs.ml.clustering.UMAP()
    # fig, embedding = UMAP.cluster(data, labels=targets, supervised=False)
    # plt.show()
