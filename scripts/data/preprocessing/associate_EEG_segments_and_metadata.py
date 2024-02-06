#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-06 08:57:48 (ywatanabe)"

import json
import random

import mngs
import numpy as np
from scripts.data.preprocessing.get_metadata import get_metadata

# from scripts.data.preprocessing.load import _load_an_eeg_file
from scripts.data.preprocessing.preprocess import (
    preprocess_raw,
    segment_processed_raw,
)

CONFIG = mngs.io.load_configs(show=False)


def associate_EEG_segments_and_metadata(
    metadata_json,
    eeg_segs,
):
    """
    Adjusts metadata based on segment index, considering the start time of each segment derived from
    the segment index, window size, and overlap factor.

    Parameters:
    - metadata_json (str): JSON string representing the file-level metadata.
    - n_segments (int): The total number of segments.

    Returns:
    - list of str: A list of JSON strings, each representing the metadata for a specific segment.
    """

    # For segments which are not defined
    if eeg_segs is None:
        return json.dumps({})

    metadata_dict = json.loads(metadata_json)

    # For datasets withtout onset_sec metadata
    if metadata_dict.get("onset_sec", None) is None:
        return [metadata_json for _ in range(len(eeg_segs))]

    # For datasets with onset_sec metadata
    segment_labels = []

    # These constants should be correctly defined or passed as parameters
    # WINDOW_SIZE_PTS = int(CONFIG["WINDOW_SIZE_SEC"] * CONFIG["TGT_FS"])
    # STEP_SIZE_PTS = int(WINDOW_SIZE_PTS / CONFIG["OVERLAP_FACTOR"])
    STEP_SIZE_SEC = int(CONFIG["WINDOW_SIZE_SEC"] / CONFIG["OVERLAP_FACTOR"])

    for ii_seg in range(len(eeg_segs)):
        start_sec = ii_seg * STEP_SIZE_SEC
        end_sec = start_sec + CONFIG["WINDOW_SIZE_SEC"]

        # Initialize segment metadata_dict with empty lists
        segment_metadata_dict = {
            "onset_sec": [],
            "offset_sec": [],
            "label": [],
        }

        # Check each label in the metadata_dict to see if it falls within the segment's time range
        for onset, offset, label in zip(
            metadata_dict["onset_sec"],
            metadata_dict["offset_sec"],
            metadata_dict["label"],
        ):
            if (onset is None) or (offset is None):
                continue
            if onset >= start_sec and offset <= end_sec:
                segment_metadata_dict["onset_sec"].append(str(onset))
                segment_metadata_dict["offset_sec"].append(str(offset))
                segment_metadata_dict["label"].append(label)

        segment_labels.append(segment_metadata_dict)

    # Convert each segment metadata_dict dictionary back to a JSON string if necessary
    return [json.dumps(label) for label in segment_labels]


if __name__ == "__main__":
    pass

    # # OpenNeuro
    # segs, meta = load_a_processed_and_segmented_eeg_file(
    #     "./data/OpenNeuro/raw/ds003474-download/sub-001/eeg/sub-001_task-ProbabilisticSelection_eeg.set"
    # )
    # print(segs.shape, len(meta))

    # # Dementia, Japan
    # segs, meta = load_a_processed_and_segmented_eeg_file(
    #     "./data/Dementia/raw/BIDS_dataset_Osk_v5.3/sub-EEGO0001/ses-010/eeg/sub-EEGO0001_ses-010_task-rest_run-01_eeg.edf"
    # )
    # print(segs.shape, len(meta))  # (3125, 14, 256) # 3125

    # # Epilepsy, Japan # fixme

    # # Epilepsy, Siena
    # lpaths_siena = mngs.gen.find(
    #     "./data/Epilepsy/raw/siena-scalp-eeg-database-1.0.0/", "f", "*.edf"
    # )

    # all_sec = 0
    # for lpath in lpaths_siena:
    #     segs, meta = load_a_processed_and_segmented_eeg_file(lpath)
    #     print(segs.shape, len(meta))  # (3125, 14, 256) # 3125
    #     lpath_sec = (
    #         len(segs) * CONFIG["WINDOW_SIZE_SEC"] / CONFIG["OVERLAP_FACTOR"]
    #     )
    #     all_sec += lpath_sec
    # print(all_sec / 60)  # 8460.55
    # all_hours = all_sec / 60 / 60  # 141 h

    # fig, axes = mngs.plt.subplots(nrows=5, sharex=True, sharey=True)
    # for ii, i_seg in enumerate(range(4, 9)):
    #     axes[ii].plot(segs[i_seg].T)
    #     axes[ii].set_title(meta[i_seg])
    # plt.show()
