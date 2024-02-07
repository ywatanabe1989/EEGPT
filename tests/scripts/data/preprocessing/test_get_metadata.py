#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-07 12:05:32 (ywatanabe)"

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import mngs
import numpy as np
from scripts.data.preprocessing.get_metadata import get_metadata
from scripts.data.preprocessing.load import find_eeg_files

CONFIG = mngs.io.load_configs(show=False)


def _test_getdata(category):
    LPATHS = np.random.permutation(find_eeg_files(category))[
        : CONFIG["N_LPATHS_TO_LOAD"]
    ]
    for lpath in LPATHS:
        print(get_metadata(lpath))
    assert True


def test_getdata_OpenNeuro():
    _test_getdata("OpenNeuro")


def test_getdata_Dementia():
    _test_getdata("Dementia")


def test_getdata_Epilepsy():
    _test_getdata("Epilepsy")


def test_getdata_MotorMovementImagery():
    _test_getdata("MotorMovementImagery")


if __name__ == "__main__":
    test_getdata_OpenNeuro()
    test_getdata_Dementia()
    test_getdata_Epilepsy()
    test_getdata_MotorMovementImagery()
