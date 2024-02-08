#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-08 18:58:14 (ywatanabe)"

import mngs
import numpy as np
from scripts.data.load.load import (
    find_eeg_files,
    load_a_processed_and_segmented_eeg_file,
)

CONFIG = mngs.io.load_configs(show=False)


def _test_load_a_processed_and_segmented_eeg_file(category):
    LPATHS = find_eeg_files(category)
    for lpath in LPATHS:
        (
            eeg_segs,
            segmented_metadata,
        ) = load_a_processed_and_segmented_eeg_file(lpath)
    assert True


def test_load_a_processed_and_segmented_eeg_file_OpenNeuro():
    _test_load_a_processed_and_segmented_eeg_file("OpenNeuro")


def test_load_a_processed_and_segmented_eeg_file_Dementia():
    _test_load_a_processed_and_segmented_eeg_file("Dementia")


def test_load_a_processed_and_segmented_eeg_file_Epilepsy():
    _test_load_a_processed_and_segmented_eeg_file("Epilepsy")


def test_load_a_processed_and_segmented_eeg_file_MotorMovementImagery():
    _test_load_a_processed_and_segmented_eeg_file("MotorMovementImagery")


if __name__ == "__main__":
    test_load_a_processed_and_segmented_eeg_file_OpenNeuro()
    test_load_a_processed_and_segmented_eeg_file_Dementia()
    test_load_a_processed_and_segmented_eeg_file_Epilepsy()
    test_load_a_processed_and_segmented_eeg_file_MotorMovementImagery()
