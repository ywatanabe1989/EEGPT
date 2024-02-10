#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-10 23:35:52 (ywatanabe)"

from pprint import pprint

import mngs
from sklearn.model_selection import train_test_split


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


def define_df_tes(df, df_tra_val_not_exclusive_yet, df_tes_not_exclusive_yet):
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
    df_tra_val, df_tra_not_exclusive_yet, df_val_not_exclusive_yet
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
        df, df_tra_val_not_exclusive_yet, df_tes_not_exclusive_yet
    )
    # Split tra vs val
    df_tra_not_exclusive_yet = filter_by_criteria(
        df_tra_val, ds_config["INCLUSION_CRITERIA_TRA"]
    )
    df_val_not_exclusive_yet = filter_by_criteria(
        df_tra_val, ds_config["INCLUSION_CRITERIA_VAL"]
    )

    df_tra, df_val = define_df_tra_and_df_val(
        df_tra_val, df_tra_not_exclusive_yet, df_val_not_exclusive_yet
    )

    df_tra = balance_tra(df_tra, ds_config)

    validate_datasets(df_tra, df_val, df_tes, ds_config)

    return df_tra, df_val, df_tes


def balance_tra(df_tra, ds_config):
    # Assuming balancing criteria is based on a specific column, e.g., 'disease_type'
    balance_col = ds_config.get("BALANCING_CRITERIA_TRA", None)
    if not balance_col:
        return df_tra  # No balancing criteria specified

    # Perform undersampling
    min_count = df_tra[balance_col].value_counts().min()
    df_balanced = df_tra.groupby(balance_col).sample(
        n=min_count, random_state=42
    )
    return df_balanced


def validate_datasets(df_tra, df_val, df_tes, ds_config):
    # Validate inclusion criteria for each set
    for df, criteria in zip(
        [df_tra, df_val, df_tes],
        [
            ds_config["INCLUSION_CRITERIA_TRA"],
            ds_config["INCLUSION_CRITERIA_VAL"],
            ds_config["INCLUSION_CRITERIA_TES"],
        ],
    ):
        for key, values in criteria.items():
            if not df[key].isin(values).all():
                raise ValueError(
                    f"Dataset does not meet inclusion criteria for {key}"
                )

    # Example for validating number of samples (if applicable)
    # if len(df_tra) > MAX_SAMPLES_TRA:
    #     raise ValueError("Training dataset exceeds maximum number of samples")

    # Validate splits (more complex, depends on exact criteria and dataset structure)
    # This is an example placeholder for how you might start this:
    # if not set(df_tra["subject_id"]).isdisjoint(df_tes["subject_id"]):
    #     raise ValueError("Training and test datasets share subjects, contrary to split criteria")

    # Balancing checks and operations would be more complex and are better handled separately
    print("Validation passed")


if __name__ == "__main__":
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
    df = mngs.io.load("./df.pkl")
    print(df.columns)

    df_tra, df_val, df_tes = main(df, ds_config)
