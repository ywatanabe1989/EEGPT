#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-01-26 12:34:38 (ywatanabe)"

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

        self.VQ = (
            None  # placeholder, this is defined in the first forward pass
        )
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

        self.activation = {}

    def forward(self, x, get_latent=False):

        if get_latent:
            self.transformer.transformer.encoder.norm.register_forward_hook(
                self.get_latent("encoder_norm")
            )

        x = self.conv_and_vq(x).transpose(1, 2)

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

    # # def to_features(self, x):
    # #     x = self.Conv2DNet(x.unsqueeze(1))
    # #     x = self.VQ(x)
    # #     return x


if __name__ == "__main__":
    m = EEGPT().cuda()
    weights_path = (
        "./scripts/train_EEGPT/2024-0126-09:25_bvYp7bIr/checkpoint.pth"
    )

    for _ in range(10):
        x = torch.randn(32, 14, 256).cuda()
        loss = m(x)
        loss.backward()
        print(loss)

    # Pretrained model
    m.load_state_dict(mngs.io.load(weights_path).state_dict())
    l = m(x, get_latent=True)

    # m.conv_and_vq(x)
    print(summary(m, x.shape))

    # =========================================================================================================
    # Layer (type:depth-idx)                                  Output Shape              Param #
    # =========================================================================================================
    # EEGPT                                                   --                        --
    # ├─Conv2DNet: 1-1                                        [64, 64]                  --
    # │    └─Conv2d: 2-1                                      [64, 16, 14, 32]          160
    # │    └─BatchNorm2d: 2-2                                 [64, 16, 14, 32]          32
    # │    └─Conv2d: 2-3                                      [64, 16, 14, 32]          2,320
    # │    └─BatchNorm2d: 2-4                                 [64, 16, 14, 32]          32
    # │    └─MaxPool2d: 2-5                                   [64, 16, 7, 16]           --
    # │    └─MaxPool2d: 2-6                                   [64, 1, 7, 16]            --
    # │    └─Conv2d: 2-7                                      [64, 32, 7, 16]           4,640
    # │    └─BatchNorm2d: 2-8                                 [64, 32, 7, 16]           64
    # │    └─Conv2d: 2-9                                      [64, 32, 7, 16]           9,248
    # │    └─BatchNorm2d: 2-10                                [64, 32, 7, 16]           64
    # │    └─MaxPool2d: 2-11                                  [64, 32, 3, 8]            --
    # │    └─MaxPool2d: 2-12                                  [64, 16, 3, 8]            --
    # │    └─Conv2d: 2-13                                     [64, 64, 3, 8]            18,496
    # │    └─BatchNorm2d: 2-14                                [64, 64, 3, 8]            128
    # │    └─Conv2d: 2-15                                     [64, 64, 3, 8]            36,928
    # │    └─BatchNorm2d: 2-16                                [64, 64, 3, 8]            128
    # │    └─MaxPool2d: 2-17                                  [64, 64, 1, 4]            --
    # │    └─MaxPool2d: 2-18                                  [64, 32, 1, 4]            --
    # ├─Conv2DNet: 1-2                                        [64, 64]                  (recursive)
    # │    └─Conv2d: 2-19                                     [64, 16, 14, 32]          (recursive)
    # │    └─BatchNorm2d: 2-20                                [64, 16, 14, 32]          (recursive)
    # │    └─Conv2d: 2-21                                     [64, 16, 14, 32]          (recursive)
    # │    └─BatchNorm2d: 2-22                                [64, 16, 14, 32]          (recursive)
    # │    └─MaxPool2d: 2-23                                  [64, 16, 7, 16]           --
    # │    └─MaxPool2d: 2-24                                  [64, 1, 7, 16]            --
    # │    └─Conv2d: 2-25                                     [64, 32, 7, 16]           (recursive)
    # │    └─BatchNorm2d: 2-26                                [64, 32, 7, 16]           (recursive)
    # │    └─Conv2d: 2-27                                     [64, 32, 7, 16]           (recursive)
    # │    └─BatchNorm2d: 2-28                                [64, 32, 7, 16]           (recursive)
    # │    └─MaxPool2d: 2-29                                  [64, 32, 3, 8]            --
    # │    └─MaxPool2d: 2-30                                  [64, 16, 3, 8]            --
    # │    └─Conv2d: 2-31                                     [64, 64, 3, 8]            (recursive)
    # │    └─BatchNorm2d: 2-32                                [64, 64, 3, 8]            (recursive)
    # │    └─Conv2d: 2-33                                     [64, 64, 3, 8]            (recursive)
    # │    └─BatchNorm2d: 2-34                                [64, 64, 3, 8]            (recursive)
    # │    └─MaxPool2d: 2-35                                  [64, 64, 1, 4]            --
    # │    └─MaxPool2d: 2-36                                  [64, 32, 1, 4]            --
    # ├─Conv2DNet: 1-3                                        [64, 64]                  (recursive)
    # │    └─Conv2d: 2-37                                     [64, 16, 14, 32]          (recursive)
    # │    └─BatchNorm2d: 2-38                                [64, 16, 14, 32]          (recursive)
    # │    └─Conv2d: 2-39                                     [64, 16, 14, 32]          (recursive)
    # │    └─BatchNorm2d: 2-40                                [64, 16, 14, 32]          (recursive)
    # │    └─MaxPool2d: 2-41                                  [64, 16, 7, 16]           --
    # │    └─MaxPool2d: 2-42                                  [64, 1, 7, 16]            --
    # │    └─Conv2d: 2-43                                     [64, 32, 7, 16]           (recursive)
    # │    └─BatchNorm2d: 2-44                                [64, 32, 7, 16]           (recursive)
    # │    └─Conv2d: 2-45                                     [64, 32, 7, 16]           (recursive)
    # │    └─BatchNorm2d: 2-46                                [64, 32, 7, 16]           (recursive)
    # │    └─MaxPool2d: 2-47                                  [64, 32, 3, 8]            --
    # │    └─MaxPool2d: 2-48                                  [64, 16, 3, 8]            --
    # │    └─Conv2d: 2-49                                     [64, 64, 3, 8]            (recursive)
    # │    └─BatchNorm2d: 2-50                                [64, 64, 3, 8]            (recursive)
    # │    └─Conv2d: 2-51                                     [64, 64, 3, 8]            (recursive)
    # │    └─BatchNorm2d: 2-52                                [64, 64, 3, 8]            (recursive)
    # │    └─MaxPool2d: 2-53                                  [64, 64, 1, 4]            --
    # │    └─MaxPool2d: 2-54                                  [64, 32, 1, 4]            --
    # ├─Conv2DNet: 1-4                                        [64, 64]                  (recursive)
    # │    └─Conv2d: 2-55                                     [64, 16, 14, 32]          (recursive)
    # │    └─BatchNorm2d: 2-56                                [64, 16, 14, 32]          (recursive)
    # │    └─Conv2d: 2-57                                     [64, 16, 14, 32]          (recursive)
    # │    └─BatchNorm2d: 2-58                                [64, 16, 14, 32]          (recursive)
    # │    └─MaxPool2d: 2-59                                  [64, 16, 7, 16]           --
    # │    └─MaxPool2d: 2-60                                  [64, 1, 7, 16]            --
    # │    └─Conv2d: 2-61                                     [64, 32, 7, 16]           (recursive)
    # │    └─BatchNorm2d: 2-62                                [64, 32, 7, 16]           (recursive)
    # │    └─Conv2d: 2-63                                     [64, 32, 7, 16]           (recursive)
    # │    └─BatchNorm2d: 2-64                                [64, 32, 7, 16]           (recursive)
    # │    └─MaxPool2d: 2-65                                  [64, 32, 3, 8]            --
    # │    └─MaxPool2d: 2-66                                  [64, 16, 3, 8]            --
    # │    └─Conv2d: 2-67                                     [64, 64, 3, 8]            (recursive)
    # │    └─BatchNorm2d: 2-68                                [64, 64, 3, 8]            (recursive)
    # │    └─Conv2d: 2-69                                     [64, 64, 3, 8]            (recursive)
    # │    └─BatchNorm2d: 2-70                                [64, 64, 3, 8]            (recursive)
    # │    └─MaxPool2d: 2-71                                  [64, 64, 1, 4]            --
    # │    └─MaxPool2d: 2-72                                  [64, 32, 1, 4]            --
    # ├─Conv2DNet: 1-5                                        [64, 64]                  (recursive)
    # │    └─Conv2d: 2-73                                     [64, 16, 14, 32]          (recursive)
    # │    └─BatchNorm2d: 2-74                                [64, 16, 14, 32]          (recursive)
    # │    └─Conv2d: 2-75                                     [64, 16, 14, 32]          (recursive)
    # │    └─BatchNorm2d: 2-76                                [64, 16, 14, 32]          (recursive)
    # │    └─MaxPool2d: 2-77                                  [64, 16, 7, 16]           --
    # │    └─MaxPool2d: 2-78                                  [64, 1, 7, 16]            --
    # │    └─Conv2d: 2-79                                     [64, 32, 7, 16]           (recursive)
    # │    └─BatchNorm2d: 2-80                                [64, 32, 7, 16]           (recursive)
    # │    └─Conv2d: 2-81                                     [64, 32, 7, 16]           (recursive)
    # │    └─BatchNorm2d: 2-82                                [64, 32, 7, 16]           (recursive)
    # │    └─MaxPool2d: 2-83                                  [64, 32, 3, 8]            --
    # │    └─MaxPool2d: 2-84                                  [64, 16, 3, 8]            --
    # │    └─Conv2d: 2-85                                     [64, 64, 3, 8]            (recursive)
    # │    └─BatchNorm2d: 2-86                                [64, 64, 3, 8]            (recursive)
    # │    └─Conv2d: 2-87                                     [64, 64, 3, 8]            (recursive)
    # │    └─BatchNorm2d: 2-88                                [64, 64, 3, 8]            (recursive)
    # │    └─MaxPool2d: 2-89                                  [64, 64, 1, 4]            --
    # │    └─MaxPool2d: 2-90                                  [64, 32, 1, 4]            --
    # ├─Conv2DNet: 1-6                                        [64, 64]                  (recursive)
    # │    └─Conv2d: 2-91                                     [64, 16, 14, 32]          (recursive)
    # │    └─BatchNorm2d: 2-92                                [64, 16, 14, 32]          (recursive)
    # │    └─Conv2d: 2-93                                     [64, 16, 14, 32]          (recursive)
    # │    └─BatchNorm2d: 2-94                                [64, 16, 14, 32]          (recursive)
    # │    └─MaxPool2d: 2-95                                  [64, 16, 7, 16]           --
    # │    └─MaxPool2d: 2-96                                  [64, 1, 7, 16]            --
    # │    └─Conv2d: 2-97                                     [64, 32, 7, 16]           (recursive)
    # │    └─BatchNorm2d: 2-98                                [64, 32, 7, 16]           (recursive)
    # │    └─Conv2d: 2-99                                     [64, 32, 7, 16]           (recursive)
    # │    └─BatchNorm2d: 2-100                               [64, 32, 7, 16]           (recursive)
    # │    └─MaxPool2d: 2-101                                 [64, 32, 3, 8]            --
    # │    └─MaxPool2d: 2-102                                 [64, 16, 3, 8]            --
    # │    └─Conv2d: 2-103                                    [64, 64, 3, 8]            (recursive)
    # │    └─BatchNorm2d: 2-104                               [64, 64, 3, 8]            (recursive)
    # │    └─Conv2d: 2-105                                    [64, 64, 3, 8]            (recursive)
    # │    └─BatchNorm2d: 2-106                               [64, 64, 3, 8]            (recursive)
    # │    └─MaxPool2d: 2-107                                 [64, 64, 1, 4]            --
    # │    └─MaxPool2d: 2-108                                 [64, 32, 1, 4]            --
    # ├─Conv2DNet: 1-7                                        [64, 64]                  (recursive)
    # │    └─Conv2d: 2-109                                    [64, 16, 14, 32]          (recursive)
    # │    └─BatchNorm2d: 2-110                               [64, 16, 14, 32]          (recursive)
    # │    └─Conv2d: 2-111                                    [64, 16, 14, 32]          (recursive)
    # │    └─BatchNorm2d: 2-112                               [64, 16, 14, 32]          (recursive)
    # │    └─MaxPool2d: 2-113                                 [64, 16, 7, 16]           --
    # │    └─MaxPool2d: 2-114                                 [64, 1, 7, 16]            --
    # │    └─Conv2d: 2-115                                    [64, 32, 7, 16]           (recursive)
    # │    └─BatchNorm2d: 2-116                               [64, 32, 7, 16]           (recursive)
    # │    └─Conv2d: 2-117                                    [64, 32, 7, 16]           (recursive)
    # │    └─BatchNorm2d: 2-118                               [64, 32, 7, 16]           (recursive)
    # │    └─MaxPool2d: 2-119                                 [64, 32, 3, 8]            --
    # │    └─MaxPool2d: 2-120                                 [64, 16, 3, 8]            --
    # │    └─Conv2d: 2-121                                    [64, 64, 3, 8]            (recursive)
    # │    └─BatchNorm2d: 2-122                               [64, 64, 3, 8]            (recursive)
    # │    └─Conv2d: 2-123                                    [64, 64, 3, 8]            (recursive)
    # │    └─BatchNorm2d: 2-124                               [64, 64, 3, 8]            (recursive)
    # │    └─MaxPool2d: 2-125                                 [64, 64, 1, 4]            --
    # │    └─MaxPool2d: 2-126                                 [64, 32, 1, 4]            --
    # ├─Conv2DNet: 1-8                                        [64, 64]                  (recursive)
    # │    └─Conv2d: 2-127                                    [64, 16, 14, 32]          (recursive)
    # │    └─BatchNorm2d: 2-128                               [64, 16, 14, 32]          (recursive)
    # │    └─Conv2d: 2-129                                    [64, 16, 14, 32]          (recursive)
    # │    └─BatchNorm2d: 2-130                               [64, 16, 14, 32]          (recursive)
    # │    └─MaxPool2d: 2-131                                 [64, 16, 7, 16]           --
    # │    └─MaxPool2d: 2-132                                 [64, 1, 7, 16]            --
    # │    └─Conv2d: 2-133                                    [64, 32, 7, 16]           (recursive)
    # │    └─BatchNorm2d: 2-134                               [64, 32, 7, 16]           (recursive)
    # │    └─Conv2d: 2-135                                    [64, 32, 7, 16]           (recursive)
    # │    └─BatchNorm2d: 2-136                               [64, 32, 7, 16]           (recursive)
    # │    └─MaxPool2d: 2-137                                 [64, 32, 3, 8]            --
    # │    └─MaxPool2d: 2-138                                 [64, 16, 3, 8]            --
    # │    └─Conv2d: 2-139                                    [64, 64, 3, 8]            (recursive)
    # │    └─BatchNorm2d: 2-140                               [64, 64, 3, 8]            (recursive)
    # │    └─Conv2d: 2-141                                    [64, 64, 3, 8]            (recursive)
    # │    └─BatchNorm2d: 2-142                               [64, 64, 3, 8]            (recursive)
    # │    └─MaxPool2d: 2-143                                 [64, 64, 1, 4]            --
    # │    └─MaxPool2d: 2-144                                 [64, 32, 1, 4]            --
    # ├─VectorQuantizer: 1-9                                  [64, 64]                  1,024
    # ├─VectorQuantizer: 1-10                                 [64, 64]                  (recursive)
    # ├─VectorQuantizer: 1-11                                 [64, 64]                  (recursive)
    # ├─VectorQuantizer: 1-12                                 [64, 64]                  (recursive)
    # ├─VectorQuantizer: 1-13                                 [64, 64]                  (recursive)
    # ├─VectorQuantizer: 1-14                                 [64, 64]                  (recursive)
    # ├─VectorQuantizer: 1-15                                 [64, 64]                  (recursive)
    # ├─VectorQuantizer: 1-16                                 [64, 64]                  (recursive)
    # ├─TransformerModel: 1-17                                [64, 1, 64]               --
    # │    └─Transformer: 2-145                               [7, 64, 64]               --
    # │    │    └─TransformerEncoder: 3-1                     [7, 64, 64]               843,584
    # │    │    └─TransformerDecoder: 3-2                     [7, 64, 64]               893,888
    # │    └─Linear: 2-146                                    [7, 64, 64]               4,160
    # =========================================================================================================
    # Total params: 1,814,896
    # Trainable params: 1,814,896
    # Non-trainable params: 0
    # Total mult-adds (G): 2.06
    # =========================================================================================================
    # Input size (MB): 0.92
    # Forward/backward pass size (MB): 250.87
    # Params size (MB): 6.66
    # Estimated Total Size (MB): 258.45
    # =========================================================================================================
