#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-03-11 16:53:03 (ywatanabe)"

import warnings

import mngs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from models.LossBalancer import LossBalancer
from models.SplitConv import SplitConv
from models.Transformer import Transformer
from models.VectorQuantizer import VectorQuantizer

CONFIG = mngs.gen.load_configs(show=False)


# class CosineSimilarityLoss(nn.Module):
#     def __init__(self):
#         super(CosineSimilarityLoss, self).__init__()
#         self.cosine_similarity = nn.CosineSimilarity(dim=1)

#     def forward(self, y_pred, y_true):
#         loss = 1 - self.cosine_similarity(y_pred, y_true)
#         return loss.mean()  # Taking the mean to aggregate over the batch


class AxiswiseDropout(nn.Module):
    def __init__(self, dropout_prob=0.5, dim=1):
        super(AxiswiseDropout, self).__init__()
        self.dropout_prob = dropout_prob
        self.dim = dim

    def forward(self, x):
        if self.training:
            sizes = [s if i == self.dim else 1 for i, s in enumerate(x.size())]
            dropout_mask = F.dropout(
                torch.ones(*sizes, device=x.device, dtype=x.dtype),
                self.dropout_prob,
                True,
            )

            # Expand the mask to the size of the input tensor and apply it
            return x * dropout_mask.expand_as(x)
        return x


def split_segments(x, window_size_pts):
    xs = [
        x[..., ii * window_size_pts : (ii + 1) * window_size_pts]
        for ii in range(x.shape[-1] // window_size_pts)
    ]
    return xs


def calc_band_powers(batched_signals, samp_rate):
    """
    Calculates the band powers of EEG signals in a batched manner.

    Parameters:
    - batched_signals (torch.Tensor): The EEG signals in batch, shape (batch_size, channels, samples).
    - samp_rate (float): The sampling rate of the EEG signals.
    - bands_dict (dict): A dictionary containing the frequency bands, e.g., {'delta': (0.5, 4), 'theta': (4, 8)}.

    Returns:
    - torch.Tensor: The calculated band powers, shape (batch_size, channels, number of bands).
    """
    bands_dict = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "lalpha": [8, 10],
        "halpha": [10, 13],
        "beta": (13, 30),
        "gamma": (30, 59),
    }

    # FFT
    fft_signals = torch.fft.rfft(batched_signals, dim=-1)
    freqs = torch.fft.rfftfreq(batched_signals.shape[-1], 1 / samp_rate)

    # Power spectrum
    power_spectrum = torch.abs(fft_signals) ** 2

    band_powers = []
    for band, (low_freq, high_freq) in bands_dict.items():
        # Find the indices of the frequency bins corresponding to the band
        band_indices = torch.where((freqs >= low_freq) & (freqs <= high_freq))[
            0
        ]

        # Sum the power spectrum within the band
        band_power = power_spectrum[..., band_indices].sum(dim=-1)
        band_powers.append(band_power.unsqueeze(-1))

    # Concatenate the band powers along the last dimension
    band_powers = torch.cat(band_powers, dim=-1)

    band_powers /= band_powers.sum(axis=-1, keepdims=True)
    return band_powers


class EEGPT(nn.Module):
    """
    EEGPT model for EEG signal processing using a GPT-like architecture.
    """

    def __init__(self, model_config):
        super().__init__()
        model_config.update(
            dict(
                window_size=32,
            )
        )
        self.model_config = model_config

        # Transformer layers
        self.transformer = Transformer(self.model_config)

        self.bn = nn.BatchNorm1d(84)

        self.feature_wise_dropout = AxiswiseDropout(dropout_prob=0.2, dim=-2)

        # self.criterion = CosineSimilarityLoss()
        self.criterion = F.mse_loss
        self.sigmoid = nn.Sigmoid()

        # Load weights
        weights_path = model_config.get("weights_path")
        if weights_path:
            self.load_weights(weights_path)
            self.freeze_all_layers()
            _ = mngs.ml.utils.check_params(self, show=True)

    # @staticmethod
    # def manhattan_dist(x, y):
    #     return (x - y).abs().sum()

    def forward(self, x, get_latent=False):
        """x.shape # [-1, 14, 256]"""

        # Calculates band powers by 1-sec segments
        xs = split_segments(x, window_size_pts=CONFIG["TGT_FS"])
        xs = [mngs.gen.to_z(x, axis=-1) for x in xs]
        x = torch.stack(
            [calc_band_powers(x, CONFIG["TGT_FS"]) for x in xs], axis=-1
        )

        # torch.Size([32, 14, 128, 9])
        # [-1, 14, 6, 9] # [batch_size, n_chs, n_bands, seq_len]
        x = x.transpose(1, 2)
        # [-1, 6, 14, 9] # [batch_size, n_bands, n_chs, seq_len]

        # Split the band powers into those from pre / post segments
        pre = x[..., :-1]
        post = x[..., -1].unsqueeze(-1)  # keeps the dimension for consistency

        # To features: 14 channels * 6 bands -> 84 features
        pre = pre.reshape(pre.shape[0], -1, pre.shape[-1])
        post = post.reshape(post.shape[0], -1, post.shape[-1])

        # Normalization in a batch using the mean and std of preceeding data
        mm_pre, ss_pre = pre.mean(axis=(0, -1), keepdims=True), pre.std(
            axis=(0, -1), keepdims=True
        )

        pre = (pre - mm_pre) / (ss_pre + 1e-5)
        post = (post - mm_pre) / (ss_pre + 1e-5)

        # Remove nan-including samples
        pre, post = self.remove_nan_inclusing_samples(pre, post)

        # Feature-wise dropout
        pre = self.feature_wise_dropout(pre)

        # Squeese post
        post = self.sigmoid(post) * 2 - 1
        post = post.squeeze()
        self.post = post.detach().cpu()  # for future reference

        # # Clipping post to stabilize its values
        # post = torch.clamp(post, min=-1, max=1)

        if get_latent:
            self.y = None
            return self.transformer(pre, get_latent=True)

        else:
            y = self.transformer(pre, get_latent=False)[..., -1]
            if torch.isnan(y).any():
                import ipdb

                ipdb.set_trace()

            if torch.isnan(post).any():
                import ipdb

                ipdb.set_trace()

            loss = self.criterion(post, y)
            self.y = y.detach().cpu()
            return loss

    def remove_nan_inclusing_samples(self, pre, post):
        is_nan_pre = pre.reshape(len(pre), -1).isnan().any(-1)
        is_nan_post = post.reshape(len(post), -1).isnan().any(-1)
        is_nan_all = is_nan_pre + is_nan_post

        pre = pre[~is_nan_all]
        post = post[~is_nan_all]
        return pre, post

    def load_weights(self, weights_path):
        state_dict = mngs.io.load(weights_path)
        self.load_state_dict(mngs.io.load(weights_path).state_dict())
        print(f"\nWeights were successfully loaded: {weights_path}")

    def freeze_all_layers(
        self,
    ):
        for param in list(self.parameters()):
            param.requires_grad = False

    def trap_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()

        return hook


def load_trained_EEGPT(ID):
    # EEGPT_model_config = mngs.io.load(
    #     f"./scripts/ml/pretrain/EEGPT/{ID}/transformer_model_config.yaml"
    # )
    EEGPT_model_config = mngs.io.load(
        f"./scripts/ml/pretrain/EEGPT/{ID}/model_config.yaml"
    )
    # try:
    #     del EEGPT_model_config["conv_vq_weights_path"]
    # except:
    #     pass
    EEGPT_model_config[
        "weights_path"
    ] = f"./scripts/ml/pretrain/EEGPT/{ID}/checkpoint.pth"
    EEGPT_model = EEGPT(EEGPT_model_config).cuda().eval()
    return EEGPT_model


if __name__ == "__main__":
    from scripts.ml.parse_model_configs import parse_model_configs

    WINDOW_SIZE_PTS = int(CONFIG["WINDOW_SIZE_SEC"] * CONFIG["TGT_FS"])

    x = torch.randn(32, 14, WINDOW_SIZE_PTS).cuda()

    model_config = parse_model_configs()

    m = EEGPT(model_config).cuda()

    loss = m(x, get_latent=False)
    l = m(x, get_latent=True)  # (-1, 32, 8)

    loss.backward()
    # loss = m(x)
