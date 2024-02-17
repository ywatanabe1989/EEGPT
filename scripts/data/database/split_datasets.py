#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-17 21:04:24 (ywatanabe)"

from pprint import pprint

import mngs
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CONFIG = mngs.gen.load_configs(show=False)

# Functions
def limit_segments_per_sub(df):
    def _limit_segments_per_sub(group):
        return group.sample(
            n=min(len(group), CONFIG["MAX_N_WINDOWS_PER_SUB"]),
            random_state=42,
        )

    df_reset = df.reset_index()
    df_limited = df_reset.groupby("subject_id").apply(_limit_segments_per_sub)
    df_limited = df_limited.set_index("id")
    return df_limited


def combine_inclusion_criteria(*multiple_criteria):
    combined_criteria = {}
    for criterion in multiple_criteria:
        for key, values in criterion.items():
            if key in combined_criteria:
                combined_criteria[key].update(values)
            else:
                combined_criteria[key] = set(values)
    for key in combined_criteria:
        combined_criteria[key] = list(combined_criteria[key])
    return combined_criteria


def filter_by_criteria(df, criteria):
    mask = pd.Series(True, index=df.index)
    for key, values in criteria.items():
        mask &= df[key].isin(values)
    return df[mask]


def define_df_tes(
    df, df_tra_val_not_exclusive_yet, df_tes_not_exclusive_yet, ds_config
):
    # find overlaps between tra-val and tes
    indi_c = list(
        set(df_tra_val_not_exclusive_yet.index)
        & set(df_tes_not_exclusive_yet.index)
    )  # conflicting

    if len(indi_c) == 0:
        return df_tra_val_not_exclusive_yet, df_tes_not_exclusive_yet

    df_c = df.loc[indi_c]  # conflicting
    df_tra_val_nc = df_tra_val_not_exclusive_yet.loc[
        ~df_tra_val_not_exclusive_yet.index.isin(indi_c)
    ]  # not conclicting
    df_tes_nc = df_tes_not_exclusive_yet.loc[
        ~df_tes_not_exclusive_yet.index.isin(indi_c)
    ]

    # Solve the found conflicts by merging labels based on the SPLIT_CRITERIA_TRA_VAL_vs_TES
    df_c["pseudo_label"] = mngs.ml.utils.merge_labels(
        *[df_c[col] for col in ds_config["SPLIT_CRITERIA_TRA_VAL_vs_TES"]]
    )
    pseudo_labels_uq = df_c["pseudo_label"].unique()
    labels_tra_val, labels_tes = train_test_split(
        pseudo_labels_uq,
        test_size=ds_config["SPLIT_RATIOS"]["tes"],
    )

    df_resolved_as_tra_val = df_c[df_c["pseudo_label"].isin(labels_tra_val)]
    df_resolved_as_tes = df_c[df_c["pseudo_label"].isin(labels_tes)]

    df_tra_val = pd.concat([df_tra_val_nc, df_resolved_as_tra_val])
    df_tes = pd.concat([df_tes_nc, df_resolved_as_tes])

    del df_tra_val["pseudo_label"], df_tes["pseudo_label"]
    return df_tra_val, df_tes


def define_df_tra_and_df_val(
    df_tra_val,
    df_tra_not_exclusive_yet,
    df_val_not_exclusive_yet,
    ds_config,
):
    # find overlaps between tra-val and tes
    indi_c = list(
        set(df_tra_not_exclusive_yet.index)
        & set(df_val_not_exclusive_yet.index)
    )  # conflicting

    if len(indi_c) == 0:
        return df_tra_not_exclusive_yet, df_val_not_exclusive_yet

    df_c = df_tra_val.loc[indi_c]  # conflicting

    df_tra_nc = df_tra_not_exclusive_yet.loc[
        ~df_tra_not_exclusive_yet.index.isin(indi_c)
    ]  # not conclicting
    df_val_nc = df_val_not_exclusive_yet.loc[
        ~df_val_not_exclusive_yet.index.isin(indi_c)
    ]

    # Solve the found conflicts by merging labels based on the SPLIT_CRITERIA_TRA_VAL_vs_TES
    df_c["pseudo_label"] = mngs.ml.utils.merge_labels(
        *[df_c[col] for col in ds_config["SPLIT_CRITERIA_TRA_vs_VAL"]]
    )
    pseudo_labels_uq = df_c["pseudo_label"].unique()
    labels_tra, labels_val = train_test_split(
        pseudo_labels_uq,
        test_size=ds_config["SPLIT_RATIOS"]["val"]
        / (
            ds_config["SPLIT_RATIOS"]["tra"] + ds_config["SPLIT_RATIOS"]["val"]
        ),
    )

    df_resolved_as_tra = df_c[df_c["pseudo_label"].isin(labels_tra)]
    df_resolved_as_val = df_c[df_c["pseudo_label"].isin(labels_val)]

    df_tra = pd.concat([df_tra_nc, df_resolved_as_tra])
    df_val = pd.concat([df_val_nc, df_resolved_as_val])

    del df_tra["pseudo_label"], df_val["pseudo_label"]
    return df_tra, df_val


def balance_tra(df_tra, ds_config):
    def under_sample(df_tra, col):
        pseudo_labels_to_use = []
        ns = []
        vals_uq = df_tra[col].unique()
        for val_uq in vals_uq:
            _pseudo_labels_to_use = mngs.gen.search(
                val_uq, df_tra["pseudo_label"].unique()
            )[1]
            _n = len(_pseudo_labels_to_use)
            pseudo_labels_to_use.append(_pseudo_labels_to_use)
            ns.append(_n)
        n_min = np.min(ns)
        pseudo_labels_to_use = np.hstack(
            [
                np.random.permutation(pltu)[:n_min]
                for pltu in pseudo_labels_to_use
            ]
        )
        return df_tra[df_tra["pseudo_label"].isin(pseudo_labels_to_use)]

    if not ds_config.get("BALANCING_CRITERIA_TRA"):
        return df_tra

    df_tra["pseudo_label"] = mngs.ml.utils.merge_labels(
        *[df_tra[col] for col in ds_config["BALANCING_CRITERIA_TRA"]]
    )
    # pseudo_labels_uq = df_tra["pseudo_label"].unique()

    for col in ds_config["BALANCING_CRITERIA_TRA"]:
        df_tra = under_sample(df_tra, col)

    print("\n" + "-" * 40)
    print(f"\nTraining data was undersampled as follows:")
    for col in ds_config["BALANCING_CRITERIA_TRA"]:
        print()
        print(col)
        print(mngs.gen.unique(df_tra[col]))
    print("\n" + "-" * 40)

    return df_tra

    # col = ds_config["BALANCING_CRITERIA_TRA"][0]

    # values, ns = np.unique(df_tra[col], return_counts=True)
    # pseudo_labels_uq

    # # Undersampling on each column iteratively. So the order is the matter.
    # for col in ds_config["BALANCING_CRITERIA_TRA"]:
    #     print(col)
    #     values, ns = np.unique(df_tra[col], return_counts=True)
    #     n_min = np.min(ns)
    #     indi_use = []
    #     for val in values:
    #         indi_use.append(
    #             np.random.permutation(
    #                 df_tra[df_tra["disease_type"] == val].index
    #             )[:n_min]
    #         )
    #     indi_use = np.hstack(indi_use)
    #     df_tra = df_tra.loc[indi_use]

    # return df_tra
    #     import ipdb

    #     ipdb.set_trace()

    # # ##
    # # # Example
    # # ds_config["BALANCING_CRITERIA_TRA"] = ["disease_type", "subject_id"]
    # # ##

    # df_tra["pseudo_label"] = mngs.ml.utils.merge_labels(
    #     *[df_tra[col] for col in ds_config["BALANCING_CRITERIA_TRA"]]
    # )
    # pseudo_labels_uq = df_tra["pseudo_label"].unique()
    # import ipdb

    # ipdb.set_trace()

    # # Assuming balancing criteria is based on a specific column, e.g., 'disease_type'

    # # Perform undersampling
    # min_count = df_tra["pseudo_label"].value_counts().min()

    # df_balanced = df_tra.groupby("pseudo_label").sample(
    #     n=min_count, random_state=42
    # )
    # # Optionally, you can drop the 'pseudo_label' column if it's no longer needed
    # df_balanced = df_balanced.drop(columns=["pseudo_label"])

    # return df_balanced


def validate_datasets(df_tra, df_val, df_tes, ds_config):
    # Validates inclusion criteria for each set
    for df_split, criteria in zip(
        [df_tra, df_val, df_tes],
        [
            ds_config["INCLUSION_CRITERIA_TRA"],
            ds_config["INCLUSION_CRITERIA_VAL"],
            ds_config["INCLUSION_CRITERIA_TES"],
        ],
    ):
        for key, values in criteria.items():
            if not df_split[key].isin(values).all():
                raise ValueError(
                    f"Dataset does not meet inclusion criteria for {key}"
                )

    # Validates exclusiveness among datasets
    is_exclusive = (
        set(df_tra.index) & set(df_val.index) & set(df_tes.index) == set()
    )
    if not is_exclusive:
        raise ValueError(
            f"Training, Validation, or Test datasets are not mutually exclusive"
        )

    # Validates N samples per subject
    for df_split, split in zip([df_tra, df_val, df_tes], ds_config["SPLITS"]):
        _, counts = np.unique(df_split["subject_id"], return_counts=True)
        if (counts > CONFIG["MAX_N_WINDOWS_PER_SUB"]).any():
            raise ValueError(
                f"The number of samples in {split.capitalize()} dataset exceeds {CONFIG['MAX_N_WINDOWS_PER_SUB']}"
            )

    # Validates test split criteria
    col_split_tra = set(
        mngs.ml.utils.merge_labels(
            *[df_tra[k] for k in ds_config["SPLIT_CRITERIA_TRA_VAL_vs_TES"]]
        )
    )
    col_split_val = set(
        mngs.ml.utils.merge_labels(
            *[df_val[k] for k in ds_config["SPLIT_CRITERIA_TRA_VAL_vs_TES"]]
        )
    )
    col_split_tes = set(
        mngs.ml.utils.merge_labels(
            *[df_tes[k] for k in ds_config["SPLIT_CRITERIA_TRA_VAL_vs_TES"]]
        )
    )
    if not set(col_split_tes) & set(col_split_tra) == set():
        raise ValueError(
            f"Test dataset is not split from training dataset based on ds_config['SPLIT_CRITERIA_TRA_VAL_vs_TES']"
        )
    if not set(col_split_tes) & set(col_split_val) == set():
        raise ValueError(
            f"Test dataset is not split from validation dataset based on ds_config['SPLIT_CRITERIA_TRA_VAL_vs_TES']"
        )

    # Validates train-vaild split criteria
    col_split_tra = set(
        mngs.ml.utils.merge_labels(
            *[df_tra[k] for k in ds_config["SPLIT_CRITERIA_TRA_vs_VAL"]]
        )
    )
    col_split_val = set(
        mngs.ml.utils.merge_labels(
            *[df_val[k] for k in ds_config["SPLIT_CRITERIA_TRA_vs_VAL"]]
        )
    )
    if not set(col_split_tra) & set(col_split_val) == set():
        raise ValueError(
            f"Validation dataset is not split from validation dataset based on ds_config['SPLIT_CRITERIA_TRA_VAL_vs_TES']"
        )

    # Balancing checks and operations would be more complex and are better handled separately
    uqs, counts = np.unique(
        df_tra[ds_config["BALANCING_CRITERIA_TRA"]], return_counts=True
    )
    if not counts.all():
        raise ValueError(
            f"Training dataset is not balanced for {ds_config['BALANCING_CRITERIA_TRA']}"
        )

    print(f"Dataset splitting is validated.")


def main(df, ds_config):
    # Combine inclusion criteria
    # Limit the number of samples per subject to reduce sampling biases
    df = limit_segments_per_sub(df)

    # Filter by criteria
    df_tes_not_exclusive_yet = filter_by_criteria(
        df, ds_config["INCLUSION_CRITERIA_TES"]
    )
    df_tra_val_not_exclusive_yet = filter_by_criteria(
        df,
        combine_inclusion_criteria(
            ds_config["INCLUSION_CRITERIA_TRA"],
            ds_config["INCLUSION_CRITERIA_VAL"],
        ),
    )

    # Split tra&val vs tes
    df_tra_val, df_tes = define_df_tes(
        df,
        df_tra_val_not_exclusive_yet,
        df_tes_not_exclusive_yet,
        ds_config,
    )
    # Split tra vs val
    df_tra_not_exclusive_yet = filter_by_criteria(
        df_tra_val, ds_config["INCLUSION_CRITERIA_TRA"]
    )
    df_val_not_exclusive_yet = filter_by_criteria(
        df_tra_val, ds_config["INCLUSION_CRITERIA_VAL"]
    )

    df_tra, df_val = define_df_tra_and_df_val(
        df_tra_val,
        df_tra_not_exclusive_yet,
        df_val_not_exclusive_yet,
        ds_config,
    )

    df_tra = balance_tra(df_tra, ds_config)

    validate_datasets(df_tra, df_val, df_tes, ds_config)

    return df_tra, df_val, df_tes


split_datasets = main

if __name__ == "__main__":
    from scripts.data.database.SQLiteDataset import SQLiteDataset

    CONFIG = mngs.io.load_configs(show=False)
    ds_config = mngs.io.load("./config/DownStream/Dementia.yaml")
    # pprint(ds_config)
    # {'BALANCING_CRITERIA_TRA': ['disease_type'],
    #  'INCLUSION_CRITERIA_TES': {'cognitive_level': ['Normal', 'Dementia'],
    #                             'disease_type': ['AD', 'DLB', 'NPH', 'HV'],
    #                             'institution': ['Nippon Life Hospital',
    #                                             'Kochi University']},
    #  'INCLUSION_CRITERIA_TRA': {'cognitive_level': ['Normal', 'Dementia'],
    #                             'disease_type': ['AD', 'DLB', 'NPH', 'HV'],
    #                             'institution': ['Osaka University']},
    #  'INCLUSION_CRITERIA_VAL': {'cognitive_level': ['Normal', 'Dementia'],
    #                             'disease_type': ['AD', 'DLB', 'NPH', 'HV'],
    #                             'institution': ['Osaka University']},
    #  'LPATH_DB': './data/Dementia/dbs/eeg_data.db',
    #  'SPLITS': ['tra', 'val', 'tes'],
    #  'SPLIT_CRITERIA_TRA_VAL_vs_TES': ['subject_id', 'institution'],
    #  'SPLIT_CRITERIA_TRA_vs_VAL': ['subject_id'],
    #  'SPLIT_RATIOS': {'tes': 0.2, 'tra': 0.6, 'val': 0.2}}

    # # for developing
    # ds_config["SPLIT_CRITERIA_TES_TRA"] = ["subject_id"]
    # ds_config["INCLUSION_CRITERIA_TES"] = {
    #     "disease_type": ["AD", "DLB", "NPH", "HV"],
    #     "cognitive_level": ["Normal", "Dementia"],
    #     "institution": ["Osaka University"],
    # }
    # Loads metadata from a sqlite database
    df = SQLiteDataset.load_meta_as_df(lpath_db=ds_config["LPATH_DB"])
    # df = mngs.io.load("./df.pkl")
    # print(df.columns)

    df_tra, df_val, df_tes = main(df, ds_config)
