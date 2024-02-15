#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-15 09:46:48 (ywatanabe)"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""Predict EEG signal of the last 0.25 seconds from the preciding EEG data using Transformer models"""

# Demo data
batch_size = 64
n_chs = 14
samp_rate = 128
len_sec = 2
len_sec_pre = 1.75

# Assuming demo_sig_torch function is defined as provided
sig = mngs.dsp.demo_sig_torch(
    batch_size=batch_size, n_chs=n_chs, samp_rate=samp_rate, len_sec=len_sec
)

# sig_pre = sig[..., : int(len_sec_pre * samp_rate)]  # Preceding sequence
# sig_post = sig[
#     ..., int(len_sec_pre * samp_rate) :
# ]  # Future sequence to predict

# sig_pre.shape
# sig_post.shape

# # Encoder
# encoder_layer = nn.TransformerEncoderLayer(d_model=n_chs, nhead=2)
# transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
# src = sig_pre.transpose(1, 2) # (64, 224, 14) # (batch_size, seq_len, n_chs)
# out = transformer_encoder(src)  # (64, 224, 14)

# # Decoder
# decoder_layer = nn.TransformerDecoderLayer(d_model=n_chs, nhead=2)
# transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
# memory = torch.rand(10, 32, 512) # memory (Tensor) – the sequence from the last layer of the encoder (required).
# # tgt = torch.rand(20, 32, 512)
# tgt = sig_post.transpose(1,2)
# out = transformer_decoder(tgt, memory)
