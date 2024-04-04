#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-03-31 07:28:23 (ywatanabe)"

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

        # self.split_segments
        self.transpose12 = mngs.nn.TransposeLayer(1, 2)

        self.transformer = Transformer(self.model_config)

        self.bn = nn.BatchNorm1d(84)

        self.feature_wise_dropout = AxiswiseDropout(dropout_prob=0.2, dim=-2)
        # self.fw_dropout = mngs.nn.AxiswiseDropout(
        #     dropout_prob=model_config.get("fw_dropout", 0.5), dim=-1
        # )

        # self.criterion = CosineSimilarityLoss()
        self.criterion = F.mse_loss
        self.sigmoid = nn.Sigmoid()

        self.load_weights(model_config.get("weights_path"))

    def forward(self, x, get_latent=False):
        """x.shape # [-1, 14, 256]"""

        xs = self.split_segments(x, window_size_pts=CONFIG["TGT_FS"])

        # Band powers as tokenization
        xs = [mngs.gen.to_z(x, axis=-1) for x in xs]
        x = torch.stack(
            [self.calc_band_powers(x, CONFIG["TGT_FS"]) for x in xs], axis=-1
        )

        # x = x.transpose(1, 2)
        x = self.transpose12(x)

        # Split the band powers into pre- and post-segments
        pre = x[..., :-1]
        post = x[..., -1].unsqueeze(-1)  # keeps the dimension for consistency

        # Reshape features: 14 channels * 6 bands -> 84 features
        pre = pre.reshape(pre.shape[0], -1, pre.shape[-1])
        post = post.reshape(post.shape[0], -1, post.shape[-1])

        # Normalization in a batch using the mean and std of the pre data
        mm_pre = pre.mean(axis=(0, -1), keepdims=True)
        ss_pre = pre.std(axis=(0, -1), keepdims=True)
        pre = (pre - mm_pre) / (ss_pre + 1e-5)
        post = (post - mm_pre) / (ss_pre + 1e-5)

        # Remove nan-including samples
        pre, post = self.remove_nan_inclusing_samples(pre, post)

        # Feature-wise dropout
        pre = self.feature_wise_dropout(pre)

        # Squash the post data
        post = self.sigmoid(post) * 2 - 1
        post = post.squeeze()

        # For future reference
        self.post = post.detach().cpu()

        if get_latent:
            self.y = None
            return self.transformer(pre, get_latent=True)

        else:
            y = self.transformer(pre, get_latent=False)[..., -1]
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

    def freeze_all_layers(
        self,
    ):
        for param in list(self.parameters()):
            param.requires_grad = False

    def trap_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()

        return hook

    @staticmethod
    def split_segments(x, window_size_pts):
        xs = [
            x[..., ii * window_size_pts : (ii + 1) * window_size_pts]
            for ii in range(x.shape[-1] // window_size_pts)
        ]
        return xs

    @staticmethod
    def calc_band_powers(batched_signals, samp_rate):
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
            band_indices = torch.where(
                (freqs >= low_freq) & (freqs <= high_freq)
            )[0]

            # Sum the power spectrum within the band
            band_power = power_spectrum[..., band_indices].sum(dim=-1)
            band_powers.append(band_power.unsqueeze(-1))

        # Concatenate the band powers along the last dimension
        band_powers = torch.cat(band_powers, dim=-1)

        band_powers /= band_powers.sum(axis=-1, keepdims=True)
        return band_powers

    def load_weights(self, weights_path):
        if weights_path:
            # state_dict = mngs.io.load(weights_path)
            self.load_state_dict(mngs.io.load(weights_path).state_dict())
            print(f"\nWeights were successfully loaded: {weights_path}")
            self.freeze_all_layers()
            _ = mngs.ml.utils.check_params(self, show=True)


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


# def split_segments(x, window_size_pts):
#     xs = [
#         x[..., ii * window_size_pts : (ii + 1) * window_size_pts]
#         for ii in range(x.shape[-1] // window_size_pts)
#     ]
#     return xs


if __name__ == "__main__":
    from scripts.ml.parse_model_configs import parse_model_configs

    WINDOW_SIZE_PTS = int(CONFIG["WINDOW_SIZE_SEC"] * CONFIG["TGT_FS"])

    x = torch.randn(32, 14, WINDOW_SIZE_PTS).cuda()

    model_config = parse_model_configs()

    model_config = {
        "tf_input_dim": 84,
        "tf_hidden_dim": 64,
        "tf_dim_feedforward": 2048,
        "tf_n_heads": 2,
        "tf_n_enc_layers": 2,
        "tf_n_dec_layers": 2,
        "tf_dropout_ratio": 0.1,
        "window_size": 32,
        "fw_dropout": 0.2,
    }

    m = EEGPT(model_config).cuda()

    loss = m(x, get_latent=False)
    l = m(x, get_latent=True)  # (-1, 32, 8)

    loss.backward()
    # loss = m(x)
