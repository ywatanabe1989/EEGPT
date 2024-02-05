#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-05 21:58:49 (ywatanabe)"

import json
import random

import mngs
import numpy as np
from scripts.utils.get_metadata import get_metadata
from scripts.utils.load_eeg_files import _load_an_eeg_file
from scripts.utils.preprocess import preprocess_raw

# from scripts.utils.handle_channels import handle_channels

CONFIG = mngs.io.load_configs(show=False)


def segment_processed_raw(processed_raw):
    assert processed_raw.info["sfreq"] == CONFIG["TGT_FS"]
    try:

        # Convert to DataFrame and transpose
        df = processed_raw.to_data_frame().set_index("time").T

        # Segmentation
        segs = mngs.dsp.crop(
            np.array(df),
            int(CONFIG["WINDOW_SIZE_SEC"] * CONFIG["TGT_FS"]),  # 256
            overlap_factor=CONFIG["OVERLAP_FACTOR"],
        )

        return segs

    except Exception as e:
        print(f"\nERROR:\nNot segmented.\n{e}")
        return None


def associate_segs_and_metadata(
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


def load_a_processed_and_segmented_eeg_file(lpath_eeg):
    def return_if_not_none(fn, arg):
        if arg is None:
            return None
        else:
            return fn(arg)

    raw = return_if_not_none(_load_an_eeg_file, lpath_eeg)

    metadata = get_metadata(lpath_eeg)

    # Process raw
    preprocessed_raw = return_if_not_none(preprocess_raw, raw)
    eeg_segs = return_if_not_none(segment_processed_raw, preprocessed_raw)

    # Process metadata
    segmented_metadata = associate_segs_and_metadata(metadata, eeg_segs)

    return eeg_segs, segmented_metadata

def test_example():
    assert True

if __name__ == "__main__":
    # OpenNeuro
    segs, meta = load_a_processed_and_segmented_eeg_file(
        "./data/OpenNeuro/raw/ds003474-download/sub-001/eeg/sub-001_task-ProbabilisticSelection_eeg.set"
    )
    print(segs.shape, len(meta))


    # Dementia, Japan
    segs, meta = load_a_processed_and_segmented_eeg_file(
        "./data/Dementia/raw/BIDS_dataset_Osk_v5.3/sub-EEGO0001/ses-010/eeg/sub-EEGO0001_ses-010_task-rest_run-01_eeg.edf"
    )
    print(segs.shape, len(meta)) # (3125, 14, 256) # 3125
    
    # Epilepsy, Japan # fixme

    # Epilepsy, Siena
    lpaths_siena = mngs.gen.find(
        "./data/Epilepsy/raw/siena-scalp-eeg-database-1.0.0/", "f", "*.edf"
    )

    all_sec = 0
    for lpath in lpaths_siena:
        segs, meta = load_a_processed_and_segmented_eeg_file(lpath)
        print(segs.shape, len(meta)) # (3125, 14, 256) # 3125        
        lpath_sec = (
            len(segs) * CONFIG["WINDOW_SIZE_SEC"] / CONFIG["OVERLAP_FACTOR"]
        )
        all_sec += lpath_sec
    print(all_sec / 60)  # 8460.55
    all_hours = all_sec / 60 / 60 # 141 h
    
    fig, axes = mngs.plt.subplots(nrows=5, sharex=True, sharey=True)
    for ii, i_seg in enumerate(range(4, 9)):
        axes[ii].plot(segs[i_seg].T)
        axes[ii].set_title(meta[i_seg])
    plt.show()

    # segs, meta = load_a_processed_and_segmented_eeg_file(lpaths_siena[1])
    # for lpath_siena in lpaths_siena:
    #     segs, meta = load_a_processed_and_segmented_eeg_file(lpath_siena)
    #  onset_sec offset_sec    label
    # 0  19.58.36   19.59.46  seizure
    # onset_sec offset_sec label

    #    patient_id   number_seizures   rec_time_minutes
    # 0        PN00                 5                198
    # 1        PN01                 2                809
    # 2        PN03                 2                752
    # 3        PN05                 3                359
    # 4        PN06                 5                722
    # 5        PN07                 1                523
    # 6        PN09                 3                410
    # 7        PN10                10               1002
    # 8        PN11                 1                145
    # 9        PN12                 4                246
    # 10       PN13                 3                519
    # 11       PN14                 4               1408
    # 12       PN16                 2                303
    # 13       PN17                 2                308
     # 128.4 h

    # ERROR:root:Error extracting metadata for ./data/Epilepsy/raw/siena-scalp-eeg-database-1.0.0/PN12/PN12-1.2.edf: list index out of range

     all_min = np.sum(
         [
 198, 
 809,     
 752, 
 359, 
 722, 
 523, 
 410, 
1002, 
 145, 
 246, 
 519, 
1408, 
 303, 
 308, 
         ])
