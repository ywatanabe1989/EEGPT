#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-05 18:44:11 (ywatanabe)"

import json
import logging
import re

import mngs

CONFIG = mngs.io.load_configs(show=False)


def _get_metadata_fake(lpath):
    return json.dumps({})


def _get_metadata_dementia_Japan(lpath_edf):
    """
    Example:
        lpaths_Osaka = mngs.gen.find("./data/Dementia/raw/BIDS_dataset_Osk_v5.3", 'f', "*.edf")
        for lpath_Osaka in lpaths_Osaka:
            print(get_metadata(lpath_Osaka))

        # {"disease_type": "AD", "cognitive_level": "MCI23", "institution": "Osaka University"}
        # {"disease_type": "unclassified", "cognitive_level": "MCI", "institution": "Osaka University"}
        # {"disease_type": "AD", "cognitive_level": "MCI23", "institution": "Osaka University"}
        # {"disease_type": "DLB", "cognitive_level": "Dementia", "institution": "Osaka University"}
        # {"disease_type": "NPH", "cognitive_level": "Dementia", "institution": "Osaka University"}
        # {"disease_type": "NPH", "cognitive_level": "MCI", "institution": "Osaka University"}
    """
    try:
        # Extract the subject ID from the EDF file path
        subject_id_match = re.search(r"sub-(\w+)", lpath_edf)
        if subject_id_match:
            subject_id = subject_id_match.group(1)
        else:
            raise ValueError("Subject ID not found in the EDF file path.")

        # Construct the base path from the EDF file path
        base_path = re.sub(r"sub-\w+/.*$", "", lpath_edf)

        # Construct the paths to the metadata files
        lpath_disease_type = f"{base_path}sub-{subject_id}/sub-{subject_id}_sessions.tsv"  # [REVISED]
        lpath_cognitive_level = f"{base_path}sub-{subject_id}/sub-{subject_id}_sessions.tsv"  # [REVISED]
        lpath_institution = lpath_edf.replace(".edf", ".json")

        # Load the metadata from the TSV files
        disease_type = mngs.io.load(lpath_disease_type)["disease_type"].iloc[0]
        cognitive_level = mngs.io.load(lpath_cognitive_level)[
            "cognitive_level"
        ].iloc[0]
        institution = mngs.io.load(lpath_institution)["InstitutionName"]

        return json.dumps(
            dict(
                disease_type=disease_type,
                cognitive_level=cognitive_level,
                institution=institution,
            )
        )
    except Exception as e:
        logging.error(f"Error extracting metadata for {lpath_edf}: {e}")
        return json.dumps({})  # Return empty JSON on error


def _get_metadata_epilepsy_Osaka(lpath_edf):  # fixme
    from glob import glob

    dirs_sub = glob("./data/Epilepsy/raw/bids_epilepsy_pros_adult_v1.1/*")
    subs = [dir_sub.split("/")[-1] for dir_sub in dirs_sub]
    for sub in subs:
        if "sub-" in sub:
            lpath = f"./data/Epilepsy/raw/bids_epilepsy_pros_adult_v1.1/{sub}/{sub}_sessions.tsv"
            df = mngs.io.load(lpath)
            print("\n" + "-" * 40)
            print(sub)
            print(df)
            print("\n" + "-" * 40)
    try:
        return None

    # mngs.io.load('data/Epilepsy/raw/bids_epilepsy_pros_adult_v1.1/participants.json')
    # mngs.io.load('data/Epilepsy/raw/bids_epilepsy_pros_adult_v1.1/participants.tsv')

    # try:
    #     # # Checks restpre
    #     lpath_pre_edf = "data/Epilepsy/raw/bids_epilepsy_pros_adult_v1.1/sub-POUN0002/ses-01/eeg/sub-POUN0002_ses-01_task-restpre_run-1_eeg.edf"
    #     lpath_pst_edf = "data/Epilepsy/raw/bids_epilepsy_pros_adult_v1.1/sub-POUN0002/ses-01/eeg/sub-POUN0002_ses-01_task-restpst_run-1_eeg.edf"

    #     # 2. It appears that healthy subjects are not included in the dataset.
    #     df = mngs.io.load(
    #             "data/Epilepsy/raw/bids_epilepsy_pros_adult_v1.1/participants.tsv"
    #         )
    #     df.columns

    #     df = pd.DataFrame(mngs.io.load(
    #             "data/Epilepsy/raw/bids_epilepsy_pros_adult_v1.1/participants.json"
    #         ))
    #     df.columns

    #     df = mngs.io.load(
    #         "data/Epilepsy/raw/bids_epilepsy_pros_adult_v1.1/participants.tsv"
    #     )
    #     df.columns

    #     df = mngs.io.load(
    #         "data/Epilepsy/raw/bids_epilepsy_pros_adult_v1.1/participants.tsv"
    #     )
    #     diagnosis_uq, counts = np.unique(df.seizure, return_counts=True)
    #     for dd, cc in zip(diagnosis_uq, counts):
    #         print(f"{dd}: n = {cc}")
    #         # focal: n = 41
    #         # general: n = 14
    #         # unknown: n = 2

    #     return json.dumps({})

    except Exception as e:
        logging.error(f"Error extracting metadata for {lpath_edf}: {e}")
        return json.dumps({})


def _get_metadata_epilepsy_Siena(lpath_edf):
    """
    Example:
        lpaths_siena = mngs.gen.find("./data/Epilepsy/raw/siena-scalp-eeg-database-1.0.0/", 'f', "*.edf")
        for lpath_siena in lpaths_siena:
            print(_get_metadata_epilepsy_Siena(lpath_siena))
            # {"onset_sec": [17.0], "offset_sec": [17.01], "label": ["seizure"]}
            # {"onset_sec": [16.2], "offset_sec": [16.21], "label": ["seizure"]}
            # {"onset_sec": [16.09], "offset_sec": [16.11], "label": ["seizure"]}
            # {"onset_sec": [7.38], "offset_sec": [7.39], "label": ["seizure"]}
            # {"onset_sec": [8.45], "offset_sec": [8.46], "label": ["seizure"]}
            # {"onset_sec": [7.55], "offset_sec": [7.55], "label": ["seizure"]}
            # {"onset_sec": [22.34], "offset_sec": [22.35], "label": ["seizure"]}
            # {"onset_sec": [16.01], "offset_sec": [16.02], "label": ["seizure"]}
            # {"onset_sec": [null], "offset_sec": [null], "label": ["seizure"]}
            # {"onset_sec": [8.1], "offset_sec": [8.11], "label": ["seizure"]}
            # {"onset_sec": [14.44], "offset_sec": [14.45], "label": ["seizure"]}
            # {"onset_sec": [null], "offset_sec": [null], "label": ["seizure"]}
            # {"onset_sec": [null], "offset_sec": [null], "label": ["seizure"]}
            # {"onset_sec": [7.45, 10.58], "offset_sec": [7.46, 10.58], "label": ["seizure", "seizure"]}
    """

    def _in_remove_comments(time_str):
        # Define a regex pattern to match comments in parentheses
        pattern = re.compile(r"\s*\([^)]*\)")
        return pattern.sub("", time_str).strip()
        return time_str

    def _in_to_float(timestr):
        """
        Converts a time string in hours.minutes.seconds format into total seconds as a float.

        Parameters:
        - timestr (str): Time string in 'hours.minutes.seconds' format.

        Returns:
        - float: Total seconds.
        """

        if not timestr:
            return 0.0  # or an appropriate default value or error handling

        parts = timestr.split(".")
        hours = int(parts[0]) * 3600
        minutes = int(parts[1]) * 60
        seconds = float(parts[2]) if len(parts) > 2 else 0.0
        return hours + minutes + seconds

    try:
        ldir, fname, _ = mngs.io.split_fpath(lpath_edf)
        subject_id = fname.split("-")[0]
        lpath_txt = ldir + f"Seizures-list-{subject_id}.txt"
        text = mngs.io.load(lpath_txt)
        seizure_times = []
        is_current_file = False

        for line in text:
            if "File name:" in line and fname in line:
                is_current_file = True
            elif "File name:" in line:
                is_current_file = False
            elif is_current_file and "Seizure start time:" in line:
                start_time = _in_to_float(
                    _in_remove_comments(line.split(":")[1].strip())
                )
            elif is_current_file and "Seizure end time:" in line:
                end_time = _in_to_float(
                    _in_remove_comments(line.split(":")[1].strip())
                )
                seizure_times.append((start_time, end_time))
                is_current_file = False  # Reset flag after finding end time

        if len(seizure_times):
            onsets_sec, offsets_sec = map(list, zip(*seizure_times))
        else:
            onsets_sec, offsets_sec = [None], [None]

        labels = ["seizure" for _ in range(len(onsets_sec))]

        return json.dumps(
            dict(
                onset_sec=onsets_sec,
                offset_sec=offsets_sec,
                label=labels,
            )
        )
    except Exception as e:
        logging.error(f"Error extracting metadata for {lpath_edf}: {e}")
        return json.dumps({})


def _get_metadata_mmi(lpath_edf):
    """
    Example:
        lpaths_mmi = mngs.gen.find("./data/MotorMovementImagery/raw/", 'f', "*.edf")
        for lpath_mmi in lpaths_mmi:
            print(get_metadata(lpath_mmi))
        # {"onset_sec": [0.0, 4.1, 8.2, 12.3, 16.4, 20.5, 24.6, 28.7, 32.8, 36.9, 41.0, 45.1, 49.2, 53.3, 57.4, 61.5, 65.6, 69.7, 73.8, 77.9, 82.0, 86.1, 90.2, 94.3, 98.4, 102.5, 106.6, 110.7, 114.8, 118.9], "offset_sec": [4.1, 8.2, 12.3, 16.4, 20.5, 24.6, 28.7, 32.8, 36.9, 41.0, 45.1, 49.2, 53.3, 57.4, 61.5, 65.6, 69.7, 73.8, 77.9, 82.0, 86.1, 90.2, 94.3, 98.4, 102.5, 106.6, 110.7, 114.8, 118.9, 123.0], "label": ["T0", "T1", "T0", "T2", "T0", "T2", "T0", "T1", "T0", "T2", "T0", "T1", "T0", "T1", "T0", "T2", "T0", "T2", "T0", "T1", "T0", "T1", "T0", "T2", "T0", "T2", "T0", "T1", "T0", "T2"]}
        # {"onset_sec": [0.0, 4.1, 8.2, 12.3, 16.4, 20.5, 24.6, 28.7, 32.8, 36.9, 41.0, 45.1, 49.2, 53.3, 57.4, 61.5, 65.6, 69.7, 73.8, 77.9, 82.0, 86.1, 90.2, 94.3, 98.4, 102.5, 106.6, 110.7, 114.8, 118.9], "offset_sec": [4.1, 8.2, 12.3, 16.4, 20.5, 24.6, 28.7, 32.8, 36.9, 41.0, 45.1, 49.2, 53.3, 57.4, 61.5, 65.6, 69.7, 73.8, 77.9, 82.0, 86.1, 90.2, 94.3, 98.4, 102.5, 106.6, 110.7, 114.8, 118.9, 123.0], "label": ["T0", "T2", "T0", "T1", "T0", "T1", "T0", "T2", "T0", "T1", "T0", "T2", "T0", "T2", "T0", "T1", "T0", "T2", "T0", "T1", "T0", "T2", "T0", "T1", "T0", "T2", "T0", "T1", "T0", "T1"]}
    """
    try:
        raw = mngs.io.load(lpath_edf, verbose=False)
        annotations = raw.annotations
        labels = [annot["description"] for annot in annotations]
        onset_times = annotations.onset
        _durations = annotations.duration
        offset_times = (onset_times + _durations).round(2)
        return json.dumps(
            dict(
                onset_sec=list(onset_times),
                offset_sec=list(offset_times),
                label=list(labels),
            )
        )
    except Exception as e:
        logging.error(f"Error extracting metadata for {lpath_edf}: {e}")
        return json.dumps({})


def _check_category(lpath):
    if "OpenNeuro" in lpath:
        return "OpenNeuro"

    if ("Dementia" in lpath) & ("BIDS_dataset" in lpath):
        return "Dementia_Japan"

    if ("Epilepsy" in lpath) & ("siena-scalp-eeg-database-1.0.0" in lpath):
        return "Epilepsy_Siena"

    if ("Epilepsy" in lpath) & ("BIDS_dataset" in lpath):
        return "Epilepsy_Japan"

    if "MotorMovementImagery" in lpath:
        return "MotorMovementImagery"


def get_metadata(lpath_edf):
    category = _check_category(lpath_edf)
    get_metadata_fn = {
        # "OpenNeuro": _get_metadata_fake,
        "Dementia_Japan": _get_metadata_dementia_Japan,
        "Epilepsy_Japan": _get_metadata_epilepsy_Osaka,
        "Epilepsy_Siena": _get_metadata_epilepsy_Siena,
        "MotorMovementImagery": _get_metadata_mmi,
    }.get(category, _get_metadata_fake)
    return get_metadata_fn(lpath_edf)


if __name__ == "__main__":
    # OpenNeuro
    print(
        get_metadata(
            "./data/OpenNeuro/raw/ds003474-download/sub-001/eeg/sub-001_task-ProbabilisticSelection_eeg.fdt"
        )
    )
    # {}

    # Dementia, Japan
    print(
        get_metadata(
            "./data/Dementia/raw/BIDS_dataset_Osk_v5.3/sub-EEGO0001/ses-010/eeg/sub-EEGO0001_ses-010_task-rest_run-01_eeg.edf"
        )
    )
    # {"disease_type": "AD", "cognitive_level": "Dementia", "institution": "Osaka University"}

    # Epilepsy, Japan # fixme

    # Epilepsy, Siena
    lpaths_siena = mngs.gen.find(
        "./data/Epilepsy/raw/siena-scalp-eeg-database-1.0.0/", "f", "*.edf"
    )
    for lpath_siena in lpaths_siena:
        print(get_metadata(lpath_siena))
    #  onset_sec offset_sec    label
    # 0  19.58.36   19.59.46  seizure
    # onset_sec offset_sec label

    # MotorMovementImagery # fixme: segmentation should be applied beforehand
    print(
        get_metadata(
            "./data/MotorMovementImagery/raw/eeg-motor-movementimagery-dataset-1.0.0/S001/S001R03.edf"
        )
    )
    #    onset_sec offset_sec label
    # 0        0.0        4.2    T0
    # 1        4.2        8.3    T2
    # 2        8.3       12.5    T0
    # 3       12.5       16.6    T1
    # 4       16.6       20.8    T0
    # 5       20.8       24.9    T1
    # 6       24.9       29.1    T0
    # 7       29.1       33.2    T2
    # 8       33.2       37.4    T0
    # 9       37.4       41.5    T2
    # 10      41.5       45.7    T0
    # 11      45.7       49.8    T1
    # 12      49.8       54.0    T0
    # 13      54.0       58.1    T1
    # 14      58.1       62.3    T0
    # 15      62.3       66.4    T2
    # 16      66.4       70.6    T0
    # 17      70.6       74.7    T1
    # 18      74.7       78.9    T0
    # 19      78.9       83.0    T2
    # 20      83.0       87.2    T0
    # 21      87.2       91.3    T2
    # 22      91.3       95.5    T0
    # 23      95.5       99.6    T1
    # 24      99.6      103.8    T0
    # 25     103.8      107.9    T1
    # 26     107.9      112.1    T0
    # 27     112.1      116.2    T2
    # 28     116.2      120.4    T0
    # 29     120.4      124.5    T1
