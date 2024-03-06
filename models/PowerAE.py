#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-27 23:46:32 (ywatanabe)"

import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic
from torch import nn
from torchsummary import summary
from torchviz import make_dot


class ConvHandler(nn.Module):
    """
    Handles convolution operations including convolution, batch normalization, activation, pooling, and their reverse operations for deconvolution.

    Parameters:
    - in_channels (int): Number of channels in the input image.
    - out_channels (int): Number of channels produced by the convolution.
    - conv_kernel_size (tuple): Size of the convolving kernel.
    - pool_kernel_size (tuple): The size of the window to take a max over.
    - verbose (bool): If True, print the shape of the tensor at this layer.

    Attributes:
    - _conv (nn.Conv2d): Convolutional layer.
    - _conv_bn (nn.BatchNorm2d): Batch normalization layer for convolutional layer.
    - _pool (nn.MaxPool2d): Pooling layer.
    - _conv_act (nn.Mish): Activation function for convolutional layer.
    - _unpool (nn.MaxUnpool2d): Unpooling layer.
    - _deconv (nn.ConvTranspose2d): Deconvolutional (transposed convolution) layer.
    - _deconv_bn (nn.BatchNorm2d): Batch normalization layer for deconvolutional layer.
    - _deconv_act (nn.Mish): Activation function for deconvolutional layer.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=16,
        conv_kernel_size=(14, 4),
        pool_kernel_size=(1, 2),
        verbose=False,
    ):
        super().__init__()

        self.verbose = verbose

        self._conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=conv_kernel_size,
        )
        self._conv_bn = nn.BatchNorm2d(out_channels)
        self._pool = nn.MaxPool2d(pool_kernel_size, return_indices=True)
        self._conv_act = nn.Mish()

        self._unpool = nn.MaxUnpool2d(pool_kernel_size)
        self._deconv = nn.ConvTranspose2d(
            in_channels=out_channels,
            out_channels=in_channels,
            kernel_size=conv_kernel_size,
        )
        self._deconv_bn = nn.BatchNorm2d(in_channels)
        self._deconv_act = nn.Mish()

    def conv(self, x):
        """
        Performs convolution operations including convolution, batch normalization, activation, and pooling.

        Parameters:
        - x (torch.Tensor): The input tensor.

        Returns:
        - x (torch.Tensor): The output tensor after convolution operations.
        """
        x = self._conv(x)
        x = self._conv_bn(x)
        x, self.indices = self._pool(x)
        x = self._conv_act(x)
        if self.verbose:
            print(x.shape)
        return x

    def deconv(self, x):
        """
        Performs deconvolution operations including unpooling, deconvolution, batch normalization, and activation.

        Parameters:
        - x (torch.Tensor): The input tensor.

        Returns:
        - x (torch.Tensor): The output tensor after deconvolution operations.
        """
        x = self._unpool(x, self.indices)
        x = self._deconv(x)
        x = self._deconv_bn(x)
        x = self._deconv_act(x)

        if self.verbose:
            print(x.shape)

        return x


class SwapLayer(nn.Module):
    """
    A layer that swaps dimensions of the input tensor.

    Parameters:
    - verbose (bool): If True, print the shape of the tensor at this layer.
    """

    def __init__(self, verbose=False):
        super().__init__()
        self.verbose = verbose

    def forward(self, x):
        """
        Swaps dimensions of the input tensor.

        Parameters:
        - x (torch.Tensor): The input tensor.

        Returns:
        - x (torch.Tensor): The tensor with swapped dimensions.
        """
        x = x.transpose(1, 2)

        if self.verbose:
            print(x.shape)

        return x


class EEGConvAE(nn.Module):
    """
    A Denoising Convolutional Autoencoder designed for EEG data preprocessing.

    This model compresses and reconstructs EEG signals with the shape of (-1, n_chs=14, seq_len=128), aiming to denoise them in the process.

    Parameters:
    - verbose (bool): Enables verbose output throughout the model if set to True.

    Attributes:
    - conv_handler_1, conv_handler_2, conv_handler_3 (ConvHandler): Convolution handlers for different stages of the model.
    - swap (SwapLayer): A layer to swap dimensions of the tensor for better processing.
    - l (torch.Tensor): Latent representation of the input data.

    Example:
        import torch
        from torchsummary import summary

        x = torch.randn(64, 14, 128).cuda()
        model = EEGConvAE(verbose=True).cuda().eval()
        y = model(x)
        l = model.l
        summary(model, x)

        print(x.shape)  # torch.Size([64, 14, 128])
        print(l.shape)  # torch.Size([64, 8, 3, 4])
        print(y.shape)  # torch.Size([64, 14, 128])

        loss = F.mse_loss(x, y)
        loss.backward()
    """

    def __init__(self, dropout_ratio=0.1, verbose=False):

        super(EEGConvAE, self).__init__()

        self.verbose = verbose
        self.dropout_ratio = dropout_ratio

        self.conv_handler_1 = ConvHandler(
            in_channels=1,
            out_channels=8,
            conv_kernel_size=(14, 1),
            pool_kernel_size=(1, 4),
            verbose=verbose,
        )
        self.conv_handler_2 = ConvHandler(
            in_channels=8,
            out_channels=8,
            conv_kernel_size=(1, 3),
            pool_kernel_size=(1, 3),
            verbose=verbose,
        )
        self.swap = SwapLayer(
            verbose=verbose,
        )
        self.conv_handler_3 = ConvHandler(
            in_channels=1,
            out_channels=8,
            conv_kernel_size=(3, 3),
            pool_kernel_size=(2, 2),
            verbose=verbose,
        )

    def forward(self, x):
        x = self.to_z(x)

        x = self.dropout_channels(x, dropout_prob=self.dropout_ratio)

        x = self.add_noise(x, alpha=0.01)

        # Convolution
        x = x.unsqueeze(1)
        x = self.conv_handler_1.conv(x)
        x = self.conv_handler_2.conv(x)
        x = self.swap(x)
        x = self.conv_handler_3.conv(x)

        # Latent features
        self.l = x.reshape(len(x), -1)

        # Deconvolution
        x = self.conv_handler_3.deconv(x)
        x = self.swap(x)
        x = self.conv_handler_2.deconv(x)
        x = self.conv_handler_1.deconv(x)
        x = x.squeeze()

        x = self.to_z(x)

        if self.verbose:
            print(f"\nCompression rate: {x.numel() / self.l.numel()}")

        return x

    def dropout_channels(self, x, dropout_prob=0.1):
        """
        Applies channel-wise dropout where dropped channels are replaced with N(0, 1) noise.

        Parameters:
            x (torch.Tensor): Input tensor.
            dropout_prob (float): Probability of dropping a channel.

        Returns:
            torch.Tensor: The tensor after applying dropout and replacement with Gaussian noise.
        """
        if not self.training:
            return x  # Only apply in training mode

        batch_size, channels, _seq_len = x.shape
        mask = (
            torch.bernoulli(
                torch.ones(batch_size, channels) * (1 - dropout_prob)
            )
            .unsqueeze(-1)
            .to(x.device)
        )
        gaussian_noise = torch.randn_like(x).to(x.device)
        x = mask * x + (1 - mask) * gaussian_noise
        return x

    def add_noise(self, x, alpha=0.1):
        if not self.training:
            return x

        if self.verbose:
            print(f"\nAdding noise...")

        return x + alpha * torch.randn_like(x).to(x.device)

    @staticmethod
    def to_z(x):
        return (x - x.mean(dim=(-2, -1), keepdims=True)) / x.std(
            dim=(-2, -1), keepdims=True
        )


if __name__ == "__main__":
    x = torch.randn(64, 14, 128).cuda()
    model = EEGConvAE(verbose=True).cuda()
    y = model(x)
    l = model.l
    summary(model, x)

    # g = make_dot(
    #     y.mean(),
    #     params=dict(
    #         model.named_parameters(),
    #     ),
    #     # show_attrs=True,
    #     # show_saved=True,
    # )
    # g.render('graph', format='png', cleanup=True)
    # # Convert to a PIL image
    # dot_png = g.render(format='png')  # This will render the graph to a PNG image
    # pil_image = Image.open(io.BytesIO(dot_png))

    print(x.shape)  # torch.Size([64, 14, 128])
    print(l.shape)  # torch.Size([64, 8, 3, 4])
    print(y.shape)  # torch.Size([64, 14, 128])
    # Compression rate: 18.7

    loss = torch.log(torch.abs(x - y) + 1e-5).mean()

    loss = F.mse_loss(x, y)
    loss.backward()

    # ==========================================================================================
    # Layer (type:depth-idx)                   Output Shape              Param #
    # ==========================================================================================
    # ├─ConvHandler: 1                         []                        --
    # |    └─Conv2d: 2-1                       [-1, 8, 1, 128]           120
    # |    └─BatchNorm2d: 2-2                  [-1, 8, 1, 128]           16
    # |    └─MaxPool2d: 2-3                    [-1, 8, 1, 32]            --
    # |    └─Mish: 2-4                         [-1, 8, 1, 32]            --
    # ├─ConvHandler: 1                         []                        --
    # |    └─Conv2d: 2-5                       [-1, 8, 1, 30]            200
    # |    └─BatchNorm2d: 2-6                  [-1, 8, 1, 30]            16
    # |    └─MaxPool2d: 2-7                    [-1, 8, 1, 10]            --
    # |    └─Mish: 2-8                         [-1, 8, 1, 10]            --
    # ├─SwapLayer: 1-1                         [-1, 1, 8, 10]            --
    # ├─ConvHandler: 1                         []                        --
    # |    └─Conv2d: 2-9                       [-1, 8, 6, 8]             80
    # |    └─BatchNorm2d: 2-10                 [-1, 8, 6, 8]             16
    # |    └─MaxPool2d: 2-11                   [-1, 8, 3, 4]             --
    # |    └─Mish: 2-12                        [-1, 8, 3, 4]             --
    # |    └─MaxUnpool2d: 2-13                 [-1, 8, 6, 8]             --
    # |    └─ConvTranspose2d: 2-14             [-1, 1, 8, 10]            73
    # |    └─BatchNorm2d: 2-15                 [-1, 1, 8, 10]            2
    # |    └─Mish: 2-16                        [-1, 1, 8, 10]            --
    # ├─SwapLayer: 1-2                         [-1, 8, 1, 10]            --
    # ├─ConvHandler: 1                         []                        --
    # |    └─MaxUnpool2d: 2-17                 [-1, 8, 1, 30]            --
    # |    └─ConvTranspose2d: 2-18             [-1, 8, 1, 32]            200
    # |    └─BatchNorm2d: 2-19                 [-1, 8, 1, 32]            16
    # |    └─Mish: 2-20                        [-1, 8, 1, 32]            --
    # ├─ConvHandler: 1                         []                        --
    # |    └─MaxUnpool2d: 2-21                 [-1, 8, 1, 128]           --
    # |    └─ConvTranspose2d: 2-22             [-1, 1, 14, 128]          113
    # |    └─BatchNorm2d: 2-23                 [-1, 1, 14, 128]          2
    # |    └─Mish: 2-24                        [-1, 1, 14, 128]          --
    # ==========================================================================================
    # Total params: 854
    # Trainable params: 854
    # Non-trainable params: 0
    # Total mult-adds (M): 0.24
