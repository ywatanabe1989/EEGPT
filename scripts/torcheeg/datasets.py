#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-03-30 16:22:47 (ywatanabe)"

from torcheeg import datasets

dataset = datasets.DEAPDataset(root_path="./data_preprocessed_python")
# ,
#                       offline_transform=transforms.Compose([
#                           transforms.BandDifferentialEntropy(),
#                           transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)
#                       ]),
#                       online_transform=transforms.ToTensor(),
#                       label_transform=transforms.Compose([
#                           transforms.Select('valence'),
#                           transforms.Binary(5.0),
#                       ]))
print(dataset[0])
