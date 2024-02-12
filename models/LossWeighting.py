#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-12 22:49:30 (ywatanabe)"

import torch
import torch.nn as nn


class LossWeighting(nn.Module):
    """Adjusts the loss for each sample based on the class frequencies to handle imbalanced datasets."""

    def __init__(self, total_classes, reduction="mean"):
        super().__init__()
        # self.register_buffer("total_classes", torch.tensor([total_classes]))
        self.total_classes = (
            total_classes  # ("total_classes", torch.tensor([total_classes]))
        )
        self.register_buffer(
            "sample_count_per_class", torch.zeros(total_classes)
        )
        self.register_buffer(
            "normalized_weights", torch.ones(total_classes) / total_classes
        )
        self.reduction = reduction

    def __call__(self, sample_losses, target_labels, is_training=True):
        device = sample_losses.device
        dtype = sample_losses.dtype
        target_labels = target_labels.to(device)

        if is_training:
            self._accumulate_class_samples(target_labels)
            self._recalculate_weights()

        # Select the weight for each sample based on its class label
        sample_weights = self.normalized_weights[target_labels]

        # Apply the weights to the sample losses
        weighted_losses = sample_losses * sample_weights
        # return weighted_losses
        if self.reduction == "mean":
            return weighted_losses.mean()
        elif self.reduction == "sum":
            return weighted_losses.sum()
        return weighted_losses

    def _accumulate_class_samples(self, target_labels):
        # Count the number of samples for each class in the batch
        unique_classes, counts = torch.unique(
            target_labels, return_counts=True
        )
        self.sample_count_per_class[unique_classes] += counts

    def _recalculate_weights(self):
        # Calculate weights as the inverse of class frequencies
        inverse_frequencies = 1.0 / (
            self.sample_count_per_class + 1e-6
        )  # Avoid division by zero
        # Normalize weights so that their sum equals the number of classes
        self.normalized_weights = (
            inverse_frequencies
            / inverse_frequencies.sum()
            * self.total_classes
        )

    # def _recalculate_weights(self):
    #     # Calculate weights as the inverse of class frequencies
    #     inverse_frequencies = 1.0 / (
    #         self.sample_count_per_class + 1e-6
    #     )  # Avoid division by zero
    #     # Normalize weights so that their sum equals the number of classes
    #     self.normalized_weights = (
    #         inverse_frequencies
    #         / inverse_frequencies.sum()
    #         * self.total_classes
    #     )
    #     self.normalized_weights = self.normalized_weights.to(
    #         device=dtype, dtype=dtype
    #     )


if __name__ == "__main__":
    total_classes = 4
    loss_weighter = LossWeighting(total_classes).cuda()
    cross_entropy = nn.CrossEntropyLoss(reduction="none")

    for _ in range(100):
        inputs = torch.randn(3, 5, requires_grad=True).cuda()
        targets = (
            torch.empty(3, dtype=torch.long).random_(total_classes).cuda()
        )
        unweighted_losses = cross_entropy(inputs, targets)
        weighted_losses = loss_weighter(
            unweighted_losses, targets, is_training=True
        )
        # weighted_loss.backward() # Uncomment during actual training
