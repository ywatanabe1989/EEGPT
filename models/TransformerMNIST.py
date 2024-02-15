#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-15 18:03:44 (ywatanabe)"

import math
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
        train_dataset, batch_size=64, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False, drop_last=True
    )
    return train_loader, val_loader


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        # Pre-compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # Make the positional encoding as long as the input sequence
        x_pe = self.pe[:, : x.size(1)]
        return x + x_pe


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_dim=196,
        hidden_dim=64,
        nheads=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
        output_dim=28,
    ):
        super(TransformerModel, self).__init__()
        self.hidden_dim = hidden_dim

        # Encoder
        self.encoder_embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(d_model=hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nheads, dim_feedforward=2048
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=2)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=2
        )

        # FC
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    @staticmethod
    def reshape(data):
        # [batch_size, seq_len n_features] to [seq_len, batch_size, n_features]
        return data.permute(1, 0, 2)

    def forward(self, pre):
        """pre: [batch_size, seq_len n_features]"""

        # Encoder
        pre = self.reshape(pre)
        pre = self.encoder_embedding(pre)
        pre = self.pos_encoder(pre)
        memory = self.transformer_encoder(pre)

        # Decoder
        tgt = pre
        out = self.transformer_decoder(tgt, memory)
        out = self.fc_out(out)
        return out[-1].unsqueeze(0)


def train_model(model, train_loader, optimizer, criterion, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            # Reshape to [batch_size, seq_len, features]
            data = data.cuda()
            pre, post = split_MNIST(data)

            optimizer.zero_grad()
            # Pass only the input to the model
            y = model(pre)
            # Calculate loss using only the predicted part
            loss = criterion(y.squeeze(), post.squeeze())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss}")


def visualize_predictions(model, data_loader):
    model.eval()
    # Randomly select an index for visualization
    chosen_idx = random.randint(0, len(data_loader.dataset) - 1)
    # Access the data directly using the chosen index
    data = data_loader.dataset[chosen_idx][0].cuda()

    pre, post = split_MNIST(data)

    y = model(pre)

    # Original image is the complete data
    original_image = data.squeeze().detach().cpu().numpy()
    # Predicted image is the first 21 rows of the original and the predicted last 7 rows
    predicted_image = (
        torch.cat((pre, y), dim=1)
        .reshape(1, 28, 28)
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_image, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(predicted_image, cmap="gray")
    prediction_start_line = (
        original_image.shape[0] * 3 // 4
    )  # Calculate the line position
    axes[1].axhline(y=prediction_start_line, color="r", linestyle="--")
    axes[1].set_title("Seq-2-Seq Generated Image")
    axes[1].axis("off")

    axes[2].imshow(
        original_image - predicted_image,
        cmap="gray",
    )
    axes[1].axhline(y=prediction_start_line, color="r", linestyle="--")
    axes[2].set_title("Difference")
    axes[2].axis("off")

    return fig
    # plt.show()


if __name__ == "__main__":
    import sys

    import matplotlib.pyplot as plt
    import mngs

    CONFIG, sys.stdout, sys.stderr, plt, cc = mngs.gen.start(sys, plt)

    dl_tra, dl_val = load_MNIST_dataloaders()
    model = TransformerModel().cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    train_model(model, dl_tra, optimizer, criterion, num_epochs=3)

    for ii in range(10):
        fig = visualize_predictions(model, dl_val)
        mngs.io.save(
            fig,
            CONFIG["SDIR"]
            + f"seq2seq_prediction/MNIST_{mngs.gen.gen_ID()}.png",
        )

    mngs.gen.close(CONFIG)
