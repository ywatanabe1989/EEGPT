#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-11 13:23:32 (ywatanabe)"

""""To tokenize EEG signals and process them in a manner similar to GPT (Generative Pre-trained Transformer) used in Natural Language Processing (NLP)"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Conv2DNet import Conv2DNet
from models.MNet import MNet
from models.VectorQuantizer import VectorQuantizer
from torch.nn import Transformer
from torchinfo import summary


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        nheads,
        num_encoder_layers,
        num_decoder_layers,
    ):
        super(TransformerModel, self).__init__()
        self.transformer = Transformer(
            d_model=input_dim,
            nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
        )
        self.fc_out = nn.Linear(input_dim, input_dim)

    def forward(self, src):
        # src: (sequence length, batch size, input dimension)
        src = src.permute(
            1, 0, 2
        )  # Transformer expects src to be of shape (S, N, E)
        output = self.transformer(src, src)
        output = self.fc_out(output)
        return output[-1, :, :].unsqueeze(1)


class EEGPT(nn.Module):
    def __init__(self):
        super().__init__()

        self.config = {}
        self.config["window_size"] = 32
        self.Conv2DNet = Conv2DNet()

        # placeholder, this is defined in the first forward pass
        self.VQ = None
        self.is_VQ_set = False

        # Model instantiation
        input_dim = 64  # Assuming each element in your sequence is of size 64
        hidden_dim = 256  # Hidden dimension size
        nheads = 4  # Number of heads in multi-head attention
        num_encoder_layers = 3  # Number of encoder layers
        num_decoder_layers = 3  # Number of decoder layers

        self.transformer = TransformerModel(
            input_dim,
            hidden_dim,
            nheads,
            num_encoder_layers,
            num_decoder_layers,
        )

        n_steps = int(self.config["window_size"] / self.config["window_size"])
        self.PE = self.generate_positional_encodings(n_steps, input_dim)

        self.activation = {}

    def forward(self, x, get_latent=False):

        if get_latent:
            self.transformer.transformer.encoder.norm.register_forward_hook(
                self.get_latent("encoder_norm")
            )

        x = self.conv_and_vq(x).transpose(1, 2)

        x += self.PE.to(x.device)

        x1 = x[:, :-1, :]  # (64, 7, 64)
        x2 = x[:, -1, :].unsqueeze(1)  # (64, 1, 64)

        y = self.transformer(x1)  # (7, 64, 64)

        if get_latent:
            l1 = (
                self.activation["encoder_norm"].transpose(0, 1).transpose(1, 2)
            )
            return l1

        loss = ((y - x2) ** 2).mean()

        return loss

    def define_VQ_if_not_defined(self, x):
        if not self.is_VQ_set:
            self.VQ = VectorQuantizer(
                num_embeddings=16, embedding_dim=x.shape[-1]
            ).to(x.device)
            self.is_VQ_set = True

    def get_latent(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()

        return hook

    def conv_and_vq(self, x):
        xs = [
            x[
                ...,
                ii
                * self.config["window_size"] : (ii + 1)
                * self.config["window_size"],
            ]
            for ii in range(x.shape[-1] // self.config["window_size"])
        ]

        xs = [self.Conv2DNet(x.unsqueeze(1)) for x in xs]

        self.define_VQ_if_not_defined(xs[0])
        xs = [self.VQ(x) for x in xs]
        x = torch.stack(xs, axis=1)  # (64, 8, 64) # seq_len = 8
        return x.transpose(1, 2)

    @classmethod
    def generate_positional_encodings(cls, length, d_model):
        PE = np.zeros((length, d_model))
        position = np.arange(0, length)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model)
        )

        PE[:, 0::2] = np.sin(position * div_term)
        PE[:, 1::2] = np.cos(position * div_term)

        PE = torch.tensor(PE).float()

        return PE


if __name__ == "__main__":
    m = EEGPT().cuda()
    # RuntimeError: Unexpected error from cudaGetDeviceCount(). Did you run some cuda functions before calling NumCudaDevices() that might have already set an error? Error 804: forward compatibility was attempted on non supported HW

    for _ in range(10):
        x = torch.randn(32, 14, 256).cuda()
        loss = m(x)
        loss.backward()
        print(loss)

    # Pretrained model
    # weights_path = (
    #     "./scripts/ml/pretrain_EEGPT/2024-02-11-07-05_dPXGGb4b/
    # )
    # m.load_state_dict(mngs.io.load(weights_path).state_dict())
    l = m(x, get_latent=True)

    # m.conv_and_vq(x)
    print(summary(m, x.shape))
