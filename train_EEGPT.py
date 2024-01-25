#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-01-25 16:16:25 (ywatanabe)"

import random

import matplotlib.pyplot as plt
import mngs
from models.EEGPT import EEGPT
from scripts.utils.SQLiteDataset import SQLiteDataset
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

CONFIG = mngs.io.load("./config/global.yaml")
WINDOW_SIZE_PTS = CONFIG["WINDOW_SIZE_SEC"] * CONFIG["TGT_FS"]

sqlite_dataset = SQLiteDataset(db_path=CONFIG["DB_PATH"])
dl = DataLoader(
    sqlite_dataset,
    batch_size=CONFIG["BATCH_SIZE"],
    shuffle=True,
    num_workers=4,
    drop_last=True,
)

model = EEGPT().cuda()
optimizer = Adam(model.parameters(), lr=float(CONFIG["LEARNING_RATE"]))
model.train()  # Set the model to training mode
losses = []
for epoch in range(CONFIG["N_EPOCHS"]):
    for i_batch, batch in enumerate(dl):
        Db, Sb, Xb = batch  # Dataset ID, Subject ID, EEG signal

        start = random.randint(0, Xb.shape[-1] - WINDOW_SIZE_PTS)
        end = start + WINDOW_SIZE_PTS

        Xb = Xb[..., start:end].float().cuda()

        optimizer.zero_grad()  # Zero the gradients before running the backward pass.
        loss = model(Xb)
        loss.backward()  # Backpropagation
        optimizer.step()  # Update the weights
        print(loss)
        losses.append(loss.detach().cpu().item())  # numpy())

sqlite_dataset.close()

fig, ax = plt.subplots()
ax.plot(losses)
mngs.io.save(fig, "./loss.png")
plt.close()

# python train_EEGPT.py
