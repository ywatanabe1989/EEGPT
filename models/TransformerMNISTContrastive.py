#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-29 12:43:52 (ywatanabe)"

import math
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from info_nce import InfoNCE, info_nce
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def split_MNIST(data):
    # (1, 8, 98) # (batch_size, seq_len, n_features)
    data = data.reshape(len(data), 4, -1)
    split_time_step = -1
    pre = data[:, :split_time_step, :]
    post = data[:, split_time_step:, :]
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

        pre = mngs.gen.to_z(pre, axis=(-1, -2))

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


def train_and_eval_model(
    model, train_loader, validation_loader, optimizer, loss_str, num_epochs=10
):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, (data, targets) in enumerate(train_loader):
            optimizer.zero_grad()

            data, targets = data.cuda(), targets.cuda()
            pre, post = split_MNIST(data)
            post = post.squeeze()
            post_pred = model(pre).squeeze()

            # Determins a positive class and found the similar data from the batch
            positive_class = targets[torch.randperm(len(targets))[0]]
            is_positive = targets == positive_class

            # Concatenates true post data and predicted data
            post_posi = post[is_positive]
            # post_nega = post[~is_positive]
            post_pred_posi = post_pred[is_positive]
            post_pred_nega = post_pred[~is_positive]

            if loss_str == "mse":
                loss = F.mse_loss(post, post_pred)

            elif loss_str == "info_nce":
                loss = info_nce(
                    query=post_posi,
                    positive_key=post_pred_posi,
                    negative_keys=post_pred_nega,
                )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss}")

        # Plots a prediction for a validation data
        LOSS_STR = {
            "mse": "MSE",
            "info_nce": "InfoNCE",
        }[loss_str]
        fig = visualize_predictions(
            model, validation_loader, title=f"{LOSS_STR}\nEpoch #{epoch:03d}"
        )
        mngs.io.save(
            fig,
            CONFIG["SDIR"]
            + f"seq2seq_prediction/{loss_str}/epoch_{epoch:03d}_MNIST_{mngs.gen.gen_ID()}.png",
        )


def visualize_predictions(model, data_loader, title=None):
    model.eval()
    # Randomly select an index for visualization
    chosen_idx = random.randint(0, len(data_loader.dataset) - 1)
    # Access the data directly using the chosen index
    data = data_loader.dataset[chosen_idx][0].cuda()

    pre, post = split_MNIST(data)

    y = model(pre)

    y = (y - y.min()) / (y.max() - y.min())  # scale to [0, 1]

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
        original_image.shape[0] * 3 // 4 - 1
    )  # Calculate the line position
    axes[1].axhline(
        y=prediction_start_line, color=CC["pink"], linestyle="--", linewidth=5
    )
    axes[1].set_title("Seq-2-Seq Generated Image")
    axes[1].axis("off")

    axes[2].imshow(
        original_image - predicted_image,
        cmap="gray",
    )
    axes[2].axhline(
        y=prediction_start_line, color=CC["pink"], linestyle="--", linewidth=5
    )
    axes[2].set_title("Difference")
    axes[2].axis("off")

    fig.suptitle(title)

    return fig


if __name__ == "__main__":
    import sys

    import matplotlib.pyplot as plt
    import mngs

    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, torch=torch, np=np, random=random
    )

    dl_tra, dl_val = load_MNIST_dataloaders()
    model = TransformerModel().cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for loss_str in ["mse", "info_nce"]:

        train_and_eval_model(
            model, dl_tra, dl_val, optimizer, loss_str, num_epochs=100
        )

    """
    ./models/TransformerMNISTContrastive.py
    """
