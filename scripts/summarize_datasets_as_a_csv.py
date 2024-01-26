#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-01-26 08:20:07 (ywatanabe)"

import re
from glob import glob

import bids
import mngs
from bids import BIDSLayout


# Functions
def from_doi_to_dataset_id_and_version(doi):
    pattern = r"openneuro\.(ds\d+)\.v(\d+\.\d+\.\d+)"
    match = re.search(pattern, doi)
    dataset_id = match.group(1)
    version = match.group(2)
    return dataset_id, version


def get_descriptions(BIDS_ROOT):
    path_description = BIDS_ROOT + "dataset_description.json"
    description = mngs.io.load(path_description)
    return pd.DataFrame(
        pd.Series(
            dict(
                DatasetName=description["Name"],
                AUTHORS=description["Authors"],
                DOI=description["DatasetDOI"],
            )
        )
    ).T


def get_participants(BIDS_ROOT):
    path_participants = BIDS_ROOT + "participants.tsv"
    return mngs.io.load(path_participants)
    # description = mngs.io.load(path_description)
    # return pd.DataFrame(
    #     pd.Series(
    #         dict(
    #             DatasetName=description["Name"],
    #             AUTHORS=description["Authors"],
    #             DOI=description["DatasetDOI"],
    #         )
    #     )
    # ).T


def get_meta(BIDS_ROOT):
    layout = BIDSLayout(BIDS_ROOT)
    entities = layout.get_entities()
    SUBJ_UQ = natsorted(entities["subject"].unique())
    META = []
    for sub in SUBJ_UQ:
        meta = mngs.io.load(
            layout.get(subject=sub, suffix="eeg", extension=".json")[0].path
        )
        for k, v in meta.items():
            meta[k] = str(v)
        meta["Subject"] = sub
        META.append(pd.Series(meta))
    META = pd.concat(META, axis=1).T  # .set_index("Subject")
    return META


def summarize(BIDS_ROOT):
    try:
        participants = get_participants(BIDS_ROOT)
        participants["Subject"] = [
            sub_id.split("-")[-1] for sub_id in participants.participant_id
        ]
        descriptions = get_descriptions(BIDS_ROOT)
        dataset_id, dataset_version = from_doi_to_dataset_id_and_version(
            descriptions.DOI.iloc[0]
        )
        meta = get_meta(BIDS_ROOT)

        out = meta.copy()
        for k in descriptions:
            out[k] = "N/A"
            out[k] = [descriptions[k].iloc[0] for _ in range(len(out))]
        out["DatasetID"] = dataset_id
        out["DatasetVersion"] = dataset_version
        out = out[
            [
                "DatasetID",
                "Subject",
                "SamplingFrequency",
                "RecordingDuration",
                "EEGChannelCount",
                "PowerLineFrequency",
                # "ManufacturersModelName",
                "SoftwareFilters",
                "AUTHORS",
                "DatasetVersion",
                "DOI",
                "DatasetName",
                "TaskName",
            ]
        ]
        out = out.set_index("Subject")
        participants = participants.set_index("Subject")
        pd.concat([out, participants], axis=1)
        print(f"\n{BIDS_ROOT} was successfully summarized.")
        return out
    except Exception as e:
        print(f"\nError:\n{BIDS_ROOT} was not summarized.\n{e}\n")
        return None


def summarize_all(BIDS_ROOTS):
    summary = pd.concat([summarize(BIDS_ROOT) for BIDS_ROOT in BIDS_ROOTS])
    summary.index = pd.Series(
        mngs.ml.utils.merge_labels(summary["DatasetID"], summary.index),
    )
    print(f"# of Subjects: {len(summary)}")
    print(f"# of Datasets: {len(summary.DatasetName.unique())}")
    print(f"Power Line Frequencies: {summary.PowerLineFrequency.unique()}")
    print(
        f"Total Available Duration: {summary.RecordingDuration.astype(float).sum() / 60 / 60} h"
    )
    return summary


if __name__ == "__main__":
    # Config
    CONFIG = mngs.io.load("./config/global.yaml")

    # Path
    BIDS_ROOTS = glob(CONFIG["DATA_DIR"] + "ds*-download/")
    summary = summarize_all(BIDS_ROOTS)
    mngs.io.save(summary, CONFIG["DATA_DIR"] + "summary.csv")

    # # of Subjects: 1420
    # # of Datasets: 17
    # Power Line Frequencies: ['60' '50' 'n/a']
    # Total Available Duration: 341.03247248263887 h

    # EOF
