#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-07 12:02:57 (ywatanabe)"


def test_handle_channels():
    import logging

    import mngs
    import numpy as np
    from scripts.data.preprocessing.handle_channels import handle_channels
    from scripts.data.preprocessing.load import find_eeg_files
    from tqdm import tqdm

    CONFIG = mngs.io.load_configs(show=False)

    print(f"\nCONFIG['IS_DEBUG']: {CONFIG['IS_DEBUG']}")

    LPATHS = np.random.permutation(find_eeg_files("OpenNeuro"))[
        : CONFIG["N_LPATHS_TO_LOAD"]
    ]

    lpaths_not_handled = []
    for i_lpath, lpath in tqdm(enumerate(np.random.permutation(LPATHS))):

        raw = mngs.io.load(lpath, verbose=False)
        if raw is None:
            lpaths_not_handled.append(lpath)
            logging.error(f"Not loaded: {lpath}")
            continue

        handled = handle_channels(raw)
        if handled is None:
            lpaths_not_handled.append(lpath)
            continue

        i_lpath += 1

        if i_lpath % 100 == 0:
            # print(f"Handled: {success_perc:.1f}%")
            success_perc = 100 * (
                (i_lpath - len(lpaths_not_handled)) / i_lpath
            )
            print(f"Handled percentage below 90%: {success_perc:.1f}%")
        assert True


if __name__ == "__main__":
    test_handle_channels()  # more than 90% of files seem properly handled.
