#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-12 21:45:25 (ywatanabe)"

import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    """
    MultiTaskLoss module for balancing losses in multi-task learning scenarios based on task uncertainties.
    Adjusts losses for multiple tasks using the uncertainty of each task as described in https://arxiv.org/abs/1705.07115.

    Attributes:
        is_regression (torch.Tensor): Indicates if a task is regression (True) or classification (False).
        log_vars (nn.Parameter): Logarithms of variances for each task, representing uncertainties.
        n_tasks (int): Number of tasks.
        reduction (str): Method to reduce the loss across tasks ('none', 'mean', 'sum').
    """

    def __init__(self, is_regression, reduction="mean"):
        super().__init__()
        self.register_buffer(
            "is_regression", torch.tensor(is_regression, dtype=torch.float32)
        )
        self.n_tasks = len(is_regression)
        self.log_vars = nn.Parameter(torch.zeros(self.n_tasks))
        self.reduction = reduction

    def forward(self, *losses):
        """
        Calculates and weights the losses of multiple tasks based on their uncertainties.

        Args:
            *losses: A variable number of tensors representing the losses of the individual tasks.

        Returns:
            torch.Tensor: The combined multi-task loss.
        """
        # Ensure the number of losses matches the number of tasks
        assert (
            len(losses) == self.n_tasks
        ), "Number of losses must match number of tasks"

        losses = [loss.mean() for loss in losses]
        losses = torch.stack(losses)  # Stack the losses to create a tensor
        stds = torch.exp(self.log_vars) ** 0.5
        precision = 1 / ((self.is_regression + 1) * (stds**2))
        weighted_losses = precision * losses + torch.log(stds)

        if self.reduction == "mean":
            return weighted_losses.mean()
        elif self.reduction == "sum":
            return weighted_losses.sum()
        return weighted_losses


if __name__ == "__main__":
    # Usage example
    # Assuming the losses for three tasks are provided in `loss0`, `loss1`, and `loss2`
    is_regression = [True, True, False]  # Task types
    multi_task_loss_module = MultiTaskLoss(is_regression)

    # Assuming `model` is a multi-task model
    params = list(model.parameters()) + list(
        multi_task_loss_module.parameters()
    )
    optimizer = torch.optim.Adam(params, lr=1e-3)

    model.train()
    multi_task_loss_module.train()

    # Calculate the multi-task loss
    multi_task_loss = multi_task_loss_module(loss0, loss1, loss2)
