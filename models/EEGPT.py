#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-08-08 11:21:45 (ywatanabe)"

import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2DNet(nn.Module):
    """68 time steps of receptive field (= 531.25 ms at 128 Hz)"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.downsample1 = nn.Conv2d(16, 32, kernel_size=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)        
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(32)                
        self.pool2 = nn.MaxPool2d(2, 2)
        self.downsample2 = nn.Conv2d(32, 64, kernel_size=1)
        self.conv5 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(64)                        
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(64)                                
        self.pool3 = nn.MaxPool2d(2, 2)
        self.downsample3 = nn.Conv2d(64, 128, kernel_size=1)
        self.conv7 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(128)                                        
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(128)                                                
        self.pool4 = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(128, 16)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) # (16, 16, 19, 128)
        out = F.relu(self.bn2(self.conv2(out))) # (16, 16, 19, 128)
        out = self.pool1(out) # (16, 16, 19, 64)
        residual = self.pool1(x) # (16, 16, 19, 64)
        out = out + residual # (16, 16, 19, 64)

        residual = out
        out = F.relu(self.bn3(self.conv3(out)))
        out = F.relu(self.bn4(self.conv4(out)))
        out = self.pool2(out)
        residual = self.pool2(residual).mean(dim=1, keepdims=True)
        out = out + residual

        residual = out
        out = F.relu(self.bn5(self.conv5(out)))
        out = F.relu(self.bn6(self.conv6(out)))
        out = self.pool3(out)
        residual = self.pool3(residual).mean(dim=1, keepdims=True)
        out = out + residual

        residual = out # (16, 64, 2, 16)
        out = F.relu(self.bn7(self.conv7(out)))
        out = F.relu(self.bn8(self.conv8(out)))
        residual = residual.mean(dim=1, keepdims=True)
        out = out + residual


        out = F.avg_pool2d(out, out.size()[2:])  # Global average pooling
        out = out.view(out.size(0), -1)  # Flatten the tensor

        out = self.fc(out)  # Fully connected layer
        return out
        


class EEGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = Conv2DNet()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=9, nhead=9)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.fc = nn.Linear(9, 1)

    def forward(self, x):
        # x: (batch_size, 19, 1280)
        x = self._normalize_time(x)
        x = x.view(x.size(0), 1, 19, 128, 10)

        x = torch.stack([self.conv(x[..., ii]) for ii in range(x.shape[-1])], dim=-1)
        x1 = x[..., :-1]  # (16, 16, 9)
        x2 = x[..., -1]  # (16, 16)

        # Apply transformer
        x1 = x1.permute(1, 0, 2)
        x1 = self.transformer_encoder(x1)
        x1 = x1.permute(1, 0, 2)

        # x1 = x1.mean(dim=-1)
        x1 = self.fc(x1).squeeze()

        return x1, x2

    @staticmethod
    def _normalize_time(x):
        return (x - x.mean(dim=-1, keepdims=True)) / x.std(dim=-1, keepdims=True)
    

if __name__ == "__main__":
    ## Demo data
    bs, n_chs, seq_len = 16, 19, 128
    x = torch.rand(bs, 1, n_chs, seq_len).float()
    model = Conv2DNet()
    y = model(x)

    bs, n_chs, seq_len = 16, 19, 1280
    x = torch.rand(bs, n_chs, seq_len).float().cuda()
    model = EEGPT().cuda()
    y1, y2 = model(x)

    # In [48]: y1.shape
    # Out[48]: torch.Size([16, 16])

    # In [49]: y2.shape
    # Out[49]: torch.Size([16, 16])

    loss = ((y1-y2)**2).sum()
    loss.backward()
