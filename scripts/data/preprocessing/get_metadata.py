#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-07 11:53:36 (ywatanabe)"

import json
import logging
import re

import mngs

CONFIG = mngs.io.load_configs(show=False)


def _check_category(lpath):
    if "OpenNeuro" in lpath:
        return "OpenNeuro"

    if ("Dementia" in lpath) & ("BIDS_dataset" in lpath):
        return "Dementia_Japan"

    if ("Epilepsy" in lpath) & ("siena-scalp-eeg-database-1.0.0" in lpath):
        return "Epilepsy_Siena"

    if ("Epilepsy" in lpath) & ("bids_epilepsy_pros_adult" in lpath):
        return "Epilepsy_Osaka"

    if "MotorMovementImagery" in lpath:
        return "MotorMovementImagery"


def get_metadata(lpath_edf):
    category = _check_category(lpath_edf)
    get_metadata_fn = {
        "Dementia_Japan": _get_metadata_dementia_Japan,
        "Epilepsy_Osaka": _get_metadata_epilepsy_Osaka,
        "Epilepsy_Siena": _get_metadata_epilepsy_Siena,
        "MotorMovementImagery": _get_metadata_mmi,
    }.get(category, _get_metadata_fake)
    return get_metadata_fn(lpath_edf)


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

        def in_get_subject_id(lpath_edf):
            # Extract the subject ID from the EDF file path
            subject_id_match = re.search(r"sub-(\w+)", lpath_edf)
            if subject_id_match:
                subject_id = subject_id_match.group(1)
            else:
                raise ValueError("Subject ID not found in the EDF file path.")
            return subject_id

        def in_get_disease_type(lpath_edf):
            subject_id = in_get_subject_id(lpath_edf)
            base_path = re.sub(r"sub-\w+/.*$", "", lpath_edf)
            lpath_disease_type = (
                f"{base_path}sub-{subject_id}/sub-{subject_id}_sessions.tsv"
            )
            # Load the metadata from the TSV files
            disease_type = mngs.io.load(lpath_disease_type)[
                "disease_type"
            ].iloc[0]
            return disease_type

        def in_get_cognitive_level(lpath_edf):
            subject_id = in_get_subject_id(lpath_edf)
            base_path = re.sub(r"sub-\w+/.*$", "", lpath_edf)
            lpath_cognitive_level = (
                f"{base_path}sub-{subject_id}/sub-{subject_id}_sessions.tsv"
            )
            key_try = "cognitive_level"
            key_except = "disease_level"
            # cognitive_level = mngs.io.load(lpath_cognitive_level)[
            #     key_try
            # ].iloc[0]

            try:
                cognitive_level = mngs.io.load(lpath_cognitive_level)[
                    key_try
                ].iloc[0]
            except Exception as e:
                # print(e)
                # print(
                #     f"{key_try} was not found in {lpath_cognitive_level}. Try to get {key_except} instead."
                # )
                cognitive_level = mngs.io.load(lpath_cognitive_level)[
                    key_except
                ].iloc[0]

            if cognitive_level == "HV":
                logging.error(
                    "Cognitive level is originally set as 'HV'. For consistency, it will be converted to 'Normal'."
                )
                cognitive_level = "Normal"

            return cognitive_level

        # working
        # def in_get_institution_from_lpath(lpath_edf):
        #     if "Nissei" in lpath_edf:
        #         return "Nissei"
        #     if "Osk" in lpath_edf:
        #         return "Osaka"
        #     if "Kochi" in lpath_edf:
        #         return "Kochi"

        def in_get_institution_from_json(lpath_edf):
            return mngs.io.load(lpath_edf.replace(".edf", ".json"))[
                "InstitutionName"
            ]

        try:
            out = json.dumps(
                dict(
                    disease_type=in_get_disease_type(lpath_edf),
                    cognitive_level=in_get_cognitive_level(lpath_edf),
                    institution=in_get_institution_from_json(lpath_edf),
                )
            )
        except Exception as e:
            print(e)
            import ipdb

            ipdb.set_trace()
        return out
    except Exception as e:
        logging.error(f"Error extracting metadata for {lpath_edf}: {e}")
        return json.dumps({})  # Return empty JSON on error


def _get_metadata_epilepsy_Osaka(lpath_edf):  # fixme
    try:
        sub = re.search(r"sub-[A-Z]+\d+", lpath_edf).group()
        dir_sub = re.search(r".*/sub-[A-Z]+\d+/", lpath_edf).group()
        lpath_ILAE = dir_sub + sub + "_sessions.tsv"
        ILAE = mngs.io.load(lpath_ILAE)["ILAE"].iloc[0]
        return json.dumps(
            dict(
                ILAE=str(ILAE),
            )
        )
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


if __name__ == "__main__":
    from scripts.data.preprocessing.load import find_eeg_files

    def _test_getdata(category):
        # OpenNeuro
        for lpath in find_eeg_files(category):
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

    test_getdata_OpenNeuro()
    test_getdata_Dementia()
    test_getdata_Epilepsy()
    test_getdata_MotorMovementImagery()
