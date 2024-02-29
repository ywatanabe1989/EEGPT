#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-28 21:19:22 (ywatanabe)"

import torch
import torch.nn as nn
import torch.nn.functional as F

# class CosineContrastiveLoss(nn.Module):
#     """
#     Compute the cosine contrastive loss across a batch of embeddings.

#     The loss is applied to all pairs of samples in the batch, which requires
#     efficient computation of pairwise cosine similarities and the application
#     of the contrastive loss formula based on those similarities.

#     Parameters:
#     - margin (float): Margin for the contrastive loss, applied to negative pairs.
#     """

#     def __init__(self, margin=0.5, reduction="mean"):
#         super(CosineContrastiveLoss, self).__init__()
#         self.margin = margin
#         self.reduction = reduction

#     def forward(self, embeddings, labels):
#         """
#         Compute the loss for the batch of embeddings.

#         embeddings: Tensor of shape (batch_size, embed_dim) containing the embeddings.
#         labels: Tensor of shape (batch_size,) containing binary labels for the batch.
#         """
#         device = embeddings.device

#         # Normalize the embeddings to have unit length
#         embeddings_norm = F.normalize(embeddings, p=2, dim=1)

#         # Compute pairwise cosine similarity
#         cosine_sim_matrix = torch.mm(embeddings_norm, embeddings_norm.t())

#         # Get the mask for positive and negative pairs
#         positive_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
#         negative_mask = ~positive_mask

#         # Separate the positive and the negative pairs
#         positive_sim = cosine_sim_matrix[positive_mask].view(-1, 1)
#         negative_sim = cosine_sim_matrix[negative_mask].view(-1, 1)

#         # Calculate losses for positive and negative pairs
#         positive_loss = 1 - positive_sim
#         negative_loss = F.relu(negative_sim - self.margin)

#         # Combine the losses
#         loss = torch.cat((positive_loss, negative_loss), dim=0).squeeze()
#         labels = (
#             torch.cat(
#                 (
#                     torch.ones(len(positive_loss)),
#                     torch.zeros(len(negative_loss)),
#                 ),
#                 dim=0,
#             )
#             .squeeze()
#             .to(device)
#         )

#         if self.reduction == "mean":
#             return loss.mean(), labels
#         else:
#             return loss, labels


class EuclideanContrastiveLoss(nn.Module):
    """
    Compute the Euclidean contrastive loss across a batch of embeddings.

    The loss encourages positive pairs to have a small Euclidean distance and
    negative pairs to have a Euclidean distance larger than a margin.

    Parameters:
    - margin (float): Margin for the contrastive loss, applied to negative pairs.
    """

    def __init__(self, margin=1.0, reduction="mean"):
        super(EuclideanContrastiveLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, embeddings, labels):
        """
        Compute the loss for the batch of embeddings.

        embeddings: Tensor of shape (batch_size, embed_dim) containing the embeddings.
        labels: Tensor of shape (batch_size,) containing binary labels for the batch.
        """
        device = embeddings.device

        # Calculate pairwise Euclidean distances
        distance_matrix = torch.cdist(embeddings, embeddings, p=2)

        # Get the mask for positive and negative pairs
        positive_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
        negative_mask = ~positive_mask

        # Separate the positive and the negative pairs
        positive_distances = distance_matrix[positive_mask].view(-1, 1)
        negative_distances = distance_matrix[negative_mask].view(-1, 1)

        # Calculate losses for positive and negative pairs
        positive_loss = (
            positive_distances  # Minimize distance for positive pairs
        )
        negative_loss = F.relu(
            self.margin - negative_distances
        )  # Maximize distance for negative pairs

        loss = -(
            (positive_loss + 1e-10).exp().sum()
            / (negative_loss + 1e-10).exp().sum()
        )

        # Combine the losses
        loss = torch.cat((positive_loss, negative_loss), dim=0).squeeze()
        labels = (
            torch.cat(
                (
                    torch.ones(len(positive_loss)),
                    torch.zeros(len(negative_loss)),
                ),
                dim=0,
            )
            .squeeze()
            .to(device)
        )

        if self.reduction == "mean":
            return loss.mean(), labels
        else:
            return loss, labels
        # # Combine the losses
        # loss = torch.cat((positive_loss, negative_loss), dim=0)

        # if self.reduction == "mean":
        #     return loss.mean()
        # else:
        #     return loss


# class CosineContrastiveLoss(nn.Module):
#     """
#     Cosine Contrastive Loss function.

#     Encourages positive pairs to have high cosine similarity and negative pairs to have low cosine similarity.
#     Positive pairs are those with a label of 1, and negative pairs have a label of 0.

#     Parameters:
#     - margin (float): Margin for negative pair loss; should be a number between 0 and 1. It's the minimum
#       cosine distance that negative pairs should ideally have.

#     Examples:
#     --------
#     model = SomeModel()
#     contrastive_loss = CosineContrastiveLoss(margin=0.5)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#     for epoch in range(num_epochs):
#         for data in dataloader:
#             img1, img2, labels = data
#             output1, output2 = model(img1), model(img2)
#             loss = contrastive_loss(output1, output2, labels)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#     """

#     def __init__(self, margin):
#         super(CosineContrastiveLoss, self).__init__()
#         self.margin = margin

#     def forward(self, output1, output2, label):
#         # Normalize the output vectors to have unit length before calculating cosine similarity
#         output1_norm = F.normalize(output1, p=2, dim=1)
#         output2_norm = F.normalize(output2, p=2, dim=1)

#         # Cosine similarity
#         cos_sim = F.cosine_similarity(output1_norm, output2_norm)

#         # Loss calculation
#         positive_loss = (
#             1 - cos_sim
#         )  # Want positive pairs to have cosine similarity of 1
#         negative_loss = torch.clamp(
#             cos_sim - self.margin, min=0
#         )  # Want negative pairs to have cosine similarity less than margin
#         loss = torch.where(label == 1, positive_loss, negative_loss)
#         loss = torch.mean(loss)

#         return loss
