#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-12 17:00:45 (ywatanabe)"

import torch
import torch.nn.functional as F


def commitment_loss(softmax_distances):
    """
    Calculates a commitment loss based on the entropy of the softmax distances.
    The loss encourages the softmax distances to be peaky, indicating a clear
    commitment to specific classes.

    Parameters:
    - softmax_distances (Tensor): The softmax representation of distances,
      with shape (batch_size, num_embeddings).

    Returns:
    - Tensor: A scalar tensor representing the commitment loss.
    """
    # Calculate the entropy of the softmax distances
    entropy = -torch.sum(
        softmax_distances * torch.log(softmax_distances + 1e-10), dim=1
    )
    # Minimize the negative entropy to encourage peaky distributions
    loss = torch.mean(entropy)
    return loss
