#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-28 10:47:18 (ywatanabe)"

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


def split_to_1_sec_segment(x, samp_rate):
    xs = [
        x[..., ii * samp_rate : (ii + 1) * samp_rate]
        for ii in range(x.shape[-1] // samp_rate)
    ]
    return xs


def calc_band_power(xs, samp_rate):
    bands_dict = {
        "delta": [0.5, 4],
        "theta": [4, 8],
        "lalpha": [8, 10],
        "halpha": [10, 13],
        "beta": [13, 32],
        "gamma": [32, 63.9],
    }

    band_powers = []
    for x in xs:
        # Apply FFT and shift zero frequency component to the center
        fft_x = torch.fft.fft(x, dim=-1)
        fft_x = torch.fft.fftshift(fft_x, dim=-1)

        # Compute power spectrum
        power_spectrum = torch.abs(fft_x) ** 2

        # Calculate frequency resolution
        freq_res = samp_rate / x.size(-1)

        # Calculate the indices for the frequency bands
        band_indices = {
            band: [int(np.floor(freq / freq_res)) for freq in freqs]
            for band, freqs in bands_dict.items()
        }

        # Calculate band power
        band_power = []
        for band, (low_idx, high_idx) in band_indices.items():
            # Sum the power spectrum values within the frequency band limits
            band_power.append(
                power_spectrum[..., low_idx:high_idx].sum(dim=-1)
            )

        band_power = torch.stack(band_power, dim=-1)
        band_powers.append(band_power)

    band_powers = torch.stack(band_powers, dim=-1)

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

        # # Convolutional Layer
        # self.split_conv = SplitConv(self.model_config)

        # # Vector Quantizer
        # self.vq = VectorQuantizer(self.model_config)

        # Transformer layers
        self.transformer = Transformer(self.model_config)

        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")

        # Load weights
        conv_vq_weights_path = model_config.get("conv_vq_weights_path")
        weights_path = model_config.get("weights_path")

        # Check if both paths are provided and not None
        if conv_vq_weights_path is not None and weights_path is not None:
            raise ValueError(
                "Both 'conv_vq_weights_path' and 'weights_path' are provided. Please specify only one."
            )

        # Proceed with the rest of the code
        if conv_vq_weights_path is not None:
            self.load_weights(conv_vq_weights_path)
            self.freeze_conv_layers()
            self.freeze_vq_layers()
            _ = mngs.ml.utils.check_params(self, show=True)

        if weights_path is not None:
            self.load_weights(weights_path)
            self.freeze_all_layers()
            _ = mngs.ml.utils.check_params(self, show=True)

    def forward(self, x, get_latent=False):
        """x.shape # [-1, 14, 256]"""

        # Calculates band powers by 1-sec segments
        xs = split_to_1_sec_segment(x, samp_rate=CONFIG["TGT_FS"])
        x = calc_band_power(xs, samp_rate=CONFIG["TGT_FS"])
        x = mngs.gen.to_z(x, axis=(1, 2))
        x = x.reshape(x.shape[0], -1, x.shape[-1])

        # Split the band powers into those from pre / post segments
        pre = x[..., :-1]
        post = x[..., -1].unsqueeze(-1)

        return self.transformer(pre, get_latent=get_latent), post

        # # Transformer
        # if get_latent:
        #     return self.transformer(
        #         pre, get_latent=True
        #     )

        # else:
        #     y = self.transformer(pre)[..., -1]
        #     # prediction_loss = F.mse_loss(y, post.squeeze())
        #     # return prediction_loss

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
    EEGPT_model_config = mngs.io.load(
        f"./scripts/ml/pretrain_EEGPT/{ID}/transformer_model_config.yaml"
    )
    try:
        del EEGPT_model_config["conv_vq_weights_path"]
    except:
        pass
    EEGPT_model_config[
        "weights_path"
    ] = f"./scripts/ml/pretrain_EEGPT/{ID}/transformer_checkpoint.pth"
    EEGPT_model = EEGPT(EEGPT_model_config).cuda().eval()
    return EEGPT_model


if __name__ == "__main__":
    from scripts.ml.parse_model_configs import parse_model_configs

    samp_rate = 128
    len_sec = 16 + 1
    x = torch.randn(32, 14, samp_rate * len_sec).cuda()

    model_config = parse_model_configs()
    # xs = split_to_1_sec_segment(x, 128)

    # out = calc_band_power(xs, samp_rate)

    # mngs.dsp.feature_extractors.amp_band(xs[0][0], samp_rate, band_str="delta")

    # model_config = mngs.io.load(
    #     "./scripts/ml/pretrain_EEGPT/FINISHED/2024Y-02M-25D-12h27m14s_9zs9/transformer_model_config.yaml"
    # )
    # model_config.pop("conv_vq_weights_path", None)
    # model_config.pop("weights_path", None)
    # weights_path = "./scripts/ml/pretrain_EEGPT/FINISHED/2024Y-02M-25D-12h27m14s_9zs9/transformer_checkpoint.pth"
    # model_config = parse_model_configs()

    # model_config["weights_path"] = weights_path

    m = EEGPT(model_config).cuda()
    # m.eval()

    l, post = m(x, get_latent=True)  # (-1, 64, 7)
    y, post = m(x, get_latent=False)  # (-1, 64, 7)

    # loss = m(x)
