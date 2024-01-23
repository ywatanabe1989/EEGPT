#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-08-07 15:44:17 (ywatanabe)"

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from glob import glob
import numpy as np
import sys
sys.path.append(".")
from models.EEGPT import EEGPT
from script.load import BidsDataset

if __name__ == "__main__":
    bids_roots = glob("./data/EEGPT_datasets/ds*")
    bids_roots = np.array(bids_roots)[[0, 1, 2, 3]]  # 1,2,3

    # Create BidsDataset
    bids_dataset = BidsDataset(bids_roots)

    # Create DataLoader
    batch_size = 128
    data_loader = DataLoader(bids_dataset, batch_size=batch_size, shuffle=True, num_workers=32)

    # Create the model
    model = EEGPT().cuda()
    # Define the loss function
    loss_fn = nn.MSELoss()

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(10000):
        for i_batch, batch in enumerate(data_loader):
            batch = batch.cuda()
            optimizer.zero_grad()

            y1, y2 = model(batch)
            loss = loss_fn(y1, y2)
            loss.backward()
            optimizer.step()

            print(f"Epoch: {epoch}, Batch: {i_batch}, Loss: {loss.item()}")
