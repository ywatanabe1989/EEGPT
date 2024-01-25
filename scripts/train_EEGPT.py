#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-01-25 17:09:49 (ywatanabe)"

import random

import matplotlib.pyplot as plt
import mngs
from models.EEGPT import EEGPT
from scripts.utils.SQLiteDataset import SQLiteDataset
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

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
# losses = []
buff = mngs.gen.listed_dict()
for i_epoch, epoch in tqdm(enumerate(range(CONFIG["N_EPOCHS"]))):
    for i_batch, batch in enumerate(dl):
        Db, Sb, Xb = batch  # Dataset ID, Subject ID, EEG signal

        start = random.randint(0, Xb.shape[-1] - WINDOW_SIZE_PTS)
        end = start + WINDOW_SIZE_PTS

        Xb = Xb[..., start:end]
        Xb = mngs.gen.to_z(Xb, axis=2)
        Xb = mngs.gen.to_z(Xb, axis=1)
        Xb = Xb.float().cuda()

        optimizer.zero_grad()  # Zero the gradients before running the backward pass.
        loss = model(Xb)
        loss.backward()  # Backpropagation
        optimizer.step()  # Update the weights

        loss = loss.detach().cpu().item()

        buff["i_epoch"].append(i_epoch)
        buff["i_batch"].append(i_batch)
        buff["loss"].append(loss)

        if i_batch % 10 == 0:
            print(
                f"Epoch# {i_epoch}, Batch #{i_batch}, Loss: {round(loss, 5)}"
            )
        # losses.append()  # numpy())

sqlite_dataset.close()

# fig, ax = plt.subplots()
# ax.plot(losses)
# mngs.io.save(fig, "./loss.png")
# plt.close()
mngs.io.save(
    buff,
    "./res/train_EEGPT.pkl",
)

# python ./scripts/train_EEGPT.py | tee ./res/logs/train_EEGPT.log
