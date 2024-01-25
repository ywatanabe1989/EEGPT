#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-01-23 12:58:36 (ywatanabe)"

import sqlite3
import torch
from torch.utils.data import Dataset, DataLoader

class SQLiteDataset(Dataset):
    def __init__(self, db_path, query, transform=None):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.query = query
        self.transform = transform
        
        # You might want to load the data into memory if it's not too large
        # Otherwise, fetch data in __getitem__
        self.cursor.execute(query)
        self.data = self.cursor.fetchall()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Fetch the data
        item = self.data[idx]
        
        # Apply any transformations (if any)
        if self.transform:
            item = self.transform(item)
        
        # Convert to tensor or appropriate format
        # Assuming item is a tuple where the first element is the input
        # and the second element is the label/target
        input_tensor = torch.tensor(item[0], dtype=torch.float32)
        target_tensor = torch.tensor(item[1], dtype=torch.long)
        
        return input_tensor, target_tensor

    def close(self):
        self.conn.close()

# Usage
db_path = './demo.db'
query = 'SELECT features, label FROM data_table'
dataset = SQLiteDataset(db_path=db_path, query=query)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate through the DataLoader in your training loop
for inputs, targets in dataloader:
    print(f"Inputs: {inputs}, Targets: {targets}")
# for inputs, targets in dataloader:
#     # Your training code here
#     pass

# Don't forget to close the database connection when done
dataset.close()

