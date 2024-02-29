#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-28 20:56:54 (ywatanabe)"

import numpy as np
import torch
import torch.nn as nn


class LossBalancer(nn.Module):
    """Balance the cost originated from an imbalanced dataset with regard to the sample sizes in an online manner."""

    def __init__(self, n_classes):
        super().__init__()
        # self.n_classes = n_classes
        self.register_buffer("n_classes", torch.tensor(n_classes).float())
        self.register_buffer("sample_counter", torch.zeros(n_classes).float())
        self.register_buffer("weights_norm", torch.ones(n_classes) / n_classes)
        self.dummy_param = nn.Parameter(torch.empty(0))

    def __call__(self, loss, Tb, is_training=True):
        self.device = self.dummy_param.device

        if is_training:  # train is True:
            self.sample_counter = self.update_counter(Tb).to(self.device)
            self.weights_norm = self.update_weights(Tb).to(self.device)

        ## Balance the Loss
        loss *= self.weights_norm
        return loss

    def update_counter(self, Tb):
        Tb = Tb.long()
        for cls, counts in zip(*np.unique(Tb.cpu(), return_counts=True)):
            self.sample_counter[cls] += counts
        return self.sample_counter

    def update_weights(self, Tb):
        weights = 0 * Tb.clone().detach().float().to(self.device)

        # for all classes
        probs = (self.sample_counter / self.sample_counter.sum()).to(
            self.device
        )
        inv_probs = (probs + 1e-8) ** (-1)

        for ii, inv_prob in enumerate(inv_probs):
            weights[Tb == ii] = inv_probs[ii]
        weights_norm = weights / weights.sum()
        return weights_norm


if __name__ == "__main__":
    # Example
    n_classes = 4
    balancer = LossBalancer(n_classes).cuda()
    xentropy_criterion = nn.CrossEntropyLoss(reduction="none")

    n_batch = 64
    for _ in range(100):
        inp = torch.randn(n_batch, 5, requires_grad=True).cuda()
        target = (
            torch.empty(n_batch, dtype=torch.long).random_(n_classes).cuda()
        )
        target[:10] = 0
        # print(target)
        loss = xentropy_criterion(inp, target).cuda()
        balanced_loss = balancer(loss, target, is_training=True)
        balanced_loss = (
            balanced_loss.mean()
        )  # Loss should be scaler to backprop
        print(target)
        print(balancer.weights_norm)
        # print(balancer.weights_norm.sum())
        # balanced_loss.backward() # when you train the model
