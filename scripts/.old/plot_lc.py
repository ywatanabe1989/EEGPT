#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-08-17 09:25:49 (ywatanabe)"

import mngs
import re
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from glob import glob

log = mngs.io.load("./res/train_0808.txt")

data = []
for l in log:
    data.append(float(l[29:]))

fig, ax = plt.subplots()
ax.plot(np.array(data))
ax.set_ylim(1e-8, 1e-0)
ax.set_yscale("log")
plt.show()


