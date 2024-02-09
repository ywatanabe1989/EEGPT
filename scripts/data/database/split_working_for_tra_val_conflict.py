#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-09 19:31:06 (ywatanabe)"

import mngs
from sklearn.model_selection import train_test_split

CONFIG = mngs.io.load_configs(show=False)
ds_config = mngs.io.load("./config/DownStream/Dementia.yaml")


def combine_inclusion_criteria(*criteria):
    combined_criteria = {}

    # Iterate over all dictionaries passed to the function
    for criterion in criteria:
        # Iterate over key-value pairs in each dictionary
        for key, values in criterion.items():
            # If the key is already in the combined dictionary, update it with new values
            if key in combined_criteria:
                combined_criteria[key].update(values)
            # If the key is not in the combined dictionary, add it
            else:
                combined_criteria[key] = set(values)

    # Convert sets back to lists for the final output
    for key in combined_criteria:
        combined_criteria[key] = list(combined_criteria[key])

    return combined_criteria


def filter_by_criteria(df, criteria):
    mask = pd.Series(True, index=df.index)
    for key, values in criteria.items():
        mask &= df[key].isin(values)
    return df[mask]


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


def get_allocations(df_orig, df_tra, df_val, df_tes):
    df = pd.DataFrame(
        False,
        index=df_orig.index,
        columns=["is_tra", "is_val", "is_tes", "unassigned"],
    ).copy()  # False bool please
    df.loc[df_tra.index, "is_tra"] = True
    df.loc[df_val.index, "is_val"] = True
    df.loc[df_tes.index, "is_tes"] = True
    df["unassigned"] = ~(df["is_tra"] + df["is_val"] + df["is_tes"])
    df = df[~df.unassigned]
    del df["unassigned"]

    print(df.sum(axis=0))
    return df


def update_conflicts(alloc):
    alloc["tra_val_tes"] = alloc["is_tra"] & alloc["is_val"] & alloc["is_tes"]
    alloc["tra_val"] = alloc["is_tra"] & alloc["is_val"]
    alloc["val_tes"] = alloc["is_val"] & alloc["is_tes"]
    alloc["tes_tra"] = alloc["is_tes"] & alloc["is_tra"]
    print(alloc.sum())
    return alloc


def resolve_conflicts_tra_val(
    df, alloc, conflicting_columns, split_criteria_tra_val
):
    """
    conflicting_columns = "tra_val"
    split_criteria_tra_val = ds_config["SPLIT_CRITERIA_TRA_VAL"]
    """
    df_conflicted = df.loc[alloc[alloc[conflicting_columns]].index]
    merged_labels = mngs.ml.utils.merge_labels(
        df_conflicted[split_criteria_tra_val]
    )
    merged_labels_uq = np.unique(merged_labels)  # len = 712

    relative_val_ratio = ds_config["SPLIT_RATIOS"]["val"] / (
        ds_config["SPLIT_RATIOS"]["tra"] + ds_config["SPLIT_RATIOS"]["val"]
    )  # 0.25

    tra_labels, val_labels = train_test_split(
        merged_labels_uq, test_size=relative_val_ratio
    )
    indi_tra = merged_labels.isin(tra_labels)
    indi_val = merged_labels.isin(val_labels)

    indi_tra = indi_tra.iloc[np.where(indi_tra)[0]].index
    indi_val = indi_val.iloc[np.where(indi_val)[0]].index

    alloc.loc[indi_tra, "is_val"] = False
    alloc.loc[indi_val, "is_tra"] = False

    alloc = update_conflicts(alloc)
    return alloc


df_orig = mngs.io.load("./df.pkl")
df = df_orig.copy()

# {'LPATH_DB': './data/Dementia/dbs/eeg_data.db',
#  'INCLUSION_CRITERIA_TRA': {'disease_type': ['AD', 'DLB', 'NPH', 'HV'],
#   'cognitive_level': ['Normal', 'Dementia'],
#   'institution': ['Osaka University']},
#  'INCLUSION_CRITERIA_VAL': {'disease_type': ['AD', 'DLB', 'NPH', 'HV'],
#   'cognitive_level': ['Normal', 'Dementia'],
#   'institution': ['Osaka University']},
#  'INCLUSION_CRITERIA_TES': {'disease_type': ['AD', 'DLB', 'NPH', 'HV'],
#   'cognitive_level': ['Normal', 'Dementia'],
#   'institution': ['Nippon Life Hospital', 'Kochi University']},
#  'SPLIT_CRITERIA_TRA_VAL': ['subject_id'],
#  'SPLIT_CRITERIA_VAL_TES': ['subject_id', 'institution'],
#  'SPLIT_RATIOS': {'tra': 0.6, 'val': 0.2, 'tes': 0.2},
#  'BALANCING_CRITERIA_TRA': ['disease_type']}

print(df.columns)
# Index(['disease_type', 'cognitive_level', 'institution', 'dataset_id',
#        'subject_id'],
#       dtype='object')

INCLUSION_CRITERIA_COMBINED = combine_inclusion_criteria(
    ds_config["INCLUSION_CRITERIA_TRA"],
    ds_config["INCLUSION_CRITERIA_VAL"],
    ds_config["INCLUSION_CRITERIA_TES"],
)
# {'disease_type': ['NPH', 'DLB', 'HV', 'AD'],
#  'cognitive_level': ['Normal', 'Dementia'],
#  'institution': ['Nippon Life Hospital',
#   'Osaka University',
#   'Kochi University']}


df_use = filter_by_criteria(df, INCLUSION_CRITERIA_COMBINED)
df_use = limit_segments_per_sub(df_use)
# mngs.gen.unique(df_use["subject_id"])
#                               Unique Elements Counts
# 0    ds_BIDS_dataset_Kochi_v1_sub-EEGK0030592    100
# 1    ds_BIDS_dataset_Kochi_v1_sub-EEGK0114750    100
# 2    ds_BIDS_dataset_Kochi_v1_sub-EEGK0128156    100
# 3    ds_BIDS_dataset_Kochi_v1_sub-EEGK0160662    100
# 4    ds_BIDS_dataset_Kochi_v1_sub-EEGK0179991    100
# ..                                        ...    ...
# 433       ds_BIDS_dataset_Osk_v5_sub-EEGO0490    100
# 434       ds_BIDS_dataset_Osk_v5_sub-EEGO0491    100
# 435       ds_BIDS_dataset_Osk_v5_sub-EEGO0492    100
# 436       ds_BIDS_dataset_Osk_v5_sub-EEGO0493    100
# 437       ds_BIDS_dataset_Osk_v5_sub-EEGO0494    100

# [438 rows x 2 columns]


df_tra = filter_by_criteria(df_use, ds_config["INCLUSION_CRITERIA_TRA"])
df_val = filter_by_criteria(df_use, ds_config["INCLUSION_CRITERIA_VAL"])
df_tes = filter_by_criteria(df_use, ds_config["INCLUSION_CRITERIA_TES"])

alloc = update_conflicts(get_allocations(df, df_tra, df_val, df_tes))
# is_tra    22000
# is_val    22000
# is_tes    21707
# dtype: int64
# is_tra         22000
# is_val         22000
# is_tes         21707
# tra_val_tes        0
# tra_val        22000
# val_tes            0
# tes_tra            0
# dtype: int64
alloc = resolve_conflicts_tra_val(
    df, alloc, "tra_val", ds_config["SPLIT_CRITERIA_TRA_VAL"]
)
# is_tra         16500
# is_val          5500
# is_tes         21707
# tra_val_tes        0
# tra_val            0
# val_tes            0
# tes_tra            0
# dtype: int64


df_tra, df_val = resolve_conflicts_tra_val(
    df, ds_config["SPLIT_CRITERIA_TRA_VAL"]
)

alloc = get_allocations(df, df_tra, df_val, df_tes)
