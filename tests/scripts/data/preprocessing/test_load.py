#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-06 09:11:30 (ywatanabe)"


def test_load_a_processed_and_segmented_eeg_file():
    from scripts.data.preprocessing.load import (
        find_eeg_files,
        load_a_processed_and_segmented_eeg_file,
    )

    LPATHS = find_eeg_files("dementia")
    segs, meta = load_a_processed_and_segmented_eeg_file(LPATHS[0])
    assert True


if __name__ == "__main__":
    test_load_a_processed_and_segmented_eeg_file()
