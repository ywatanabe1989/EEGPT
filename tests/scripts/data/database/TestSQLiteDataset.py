#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-07 15:01:59 (ywatanabe)"

from scripts.data.database.SQLiteDataset import SQLiteDataset


def test_SQLiteDataset():
    import os
    import random
    import sys
    from datetime import datetime

    import matplotlib.pyplot as plt
    import mngs

    CONFIG, sys.stdout, sys.stderr, plt, cc = mngs.gen.start(sys, plt)
    WINDOW_SIZE_PTS = CONFIG["WINDOW_SIZE_SEC"] * CONFIG["TGT_FS"]

    # Construct dataset
    lpath_db = "./data/Dementia/dbs/eeg_data.db"
    meta_all = fetch_metadata_of_db(lpath_db)
    filter_criteria = {
        "disease_type": ["AD", "DLB", "NPH", "HV"],
        "cognitive_level": ["Normal", "Dementia", "MCI", "MCI23"],
        # "institution": ["Osaka University"],  # "Osaka",
    }
    meta_allow = apply_filter_criteria(meta_all, filter_criteria)
    ds = SQLiteDataset(
        lpath_db=lpath_db,
        filter_criteria=meta_allow,
        show=False,
    )
    assert True


if __name__ == "__main__":
    test_SQLiteDataset()
