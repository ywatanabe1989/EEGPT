#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-07 18:59:02 (ywatanabe)"


# def split_by_criteria(dataset, split_ratios, split_criteria):
#     """
#     Splits the dataset based on provided criteria ensuring that each segment
#     (training, validation, testing) contains exclusive groups based on split_criteria.
#     """
#     subsets = {}
#     for criteria, _ in split_criteria.items():
#         unique_values = dataset.get_unique_values(criteria)
#         np.random.shuffle(unique_values)  # Randomize

#         # Calculate split sizes
#         num_train = int(len(unique_values) * split_ratios["train"])
#         num_val = int(len(unique_values) * split_ratios["val"])

#         # Split values
#         train_values = unique_values[:num_train]
#         val_values = unique_values[num_train : num_train + num_val]
#         test_values = unique_values[num_train + num_val :]

#         # Fetch indices for each subset
#         indices_train = dataset.fetch_indices_by_criteria(
#             criteria, train_values
#         )
#         indices_val = dataset.fetch_indices_by_criteria(criteria, val_values)
#         indices_test = dataset.fetch_indices_by_criteria(criteria, test_values)

#         subsets["train"] = torch.utils.data.Subset(dataset, indices_train)
#         subsets["val"] = torch.utils.data.Subset(dataset, indices_val)
#         subsets["test"] = torch.utils.data.Subset(dataset, indices_test)

#     return subsets["train"], subsets["val"], subsets["test"]


# def undersample_dataset(dataset, criteria):
#     """
#     Under-samples the dataset based on the distribution of a specified criterion.

#     Parameters:
#     - dataset: Instance of SQLiteDataset.
#     - criteria: Criterion to base under-sampling on (e.g., 'disease_type').

#     Returns:
#     - A new dataset instance that is under-sampled based on the criterion.
#     """
#     # Assuming a method to count instances per criteria value exists:
#     counts = dataset.count_instances_by_criteria(criteria)
#     min_count = min(counts.values())

#     # Assuming a method to fetch indices per criteria value exists:
#     balanced_indices = []
#     for value, _ in counts.items():
#         indices = dataset.fetch_indices_by_criteria(criteria, [value])
#         np.random.shuffle(indices)
#         balanced_indices.extend(indices[:min_count])  #


# def split_by_criteria(dataset, split_ratios, split_criteria):
#     """
#     Splits the dataset based on provided criteria ensuring that each segment
#     (training, validation, testing) contains exclusive groups based on split_criteria.

#     Parameters:
#     - dataset: Instance of SQLiteDataset.
#     - split_ratios: Dict with 'train', 'val', 'test' ratios.
#     - split_criteria: Criteria used for exclusive splitting (e.g., 'subject_id', 'institution').

#     Returns:
#     - Dict containing subsets for 'train', 'val', 'test'.
#     """
#     exclusive_groups = dataset.get_unique_values(
#         split_criteria
#     )  # Implement this method
#     np.random.shuffle(
#         exclusive_groups
#     )  # Shuffle to randomize group assignment

#     # Calculate the number of groups for each set
#     n_train = int(len(exclusive_groups) * split_ratios["train"])
#     n_val = int(len(exclusive_groups) * split_ratios["val"])

#     # Assign groups to each set
#     groups_train = exclusive_groups[:n_train]
#     groups_val = exclusive_groups[n_train : n_train + n_val]
#     groups_test = exclusive_groups[n_train + n_val :]

#     # Fetch indices for each set
#     indices_train = dataset.fetch_indices_by_groups(
#         split_criteria, groups_train
#     )  # Implement
#     indices_val = dataset.fetch_indices_by_groups(
#         split_criteria, groups_val
#     )  # Implement
#     indices_test = dataset.fetch_indices_by_groups(
#         split_criteria, groups_test
#     )  # Implement

#     # Create subsets
#     subsets = {
#         "train": torch.utils.data.Subset(dataset, indices_train),
#         "val": torch.utils.data.Subset(dataset, indices_val),
#         "test": torch.utils.data.Subset(dataset, indices_test),
#     }
#     return subsets


# import mngs
# from scripts.data.database.SQLiteDataset import (
#     SQLiteDataset,
#     apply_filter_criteria,
#     fetch_metadata_of_db,
# )

# CONFIG, sys.stdout, sys.stderr, plt, cc = mngs.gen.start(sys, plt)
# WINDOW_SIZE_PTS = CONFIG["WINDOW_SIZE_SEC"] * CONFIG["TGT_FS"]

# lpath_db = "./data/Dementia/dbs/eeg_data.db"
# meta_all = fetch_metadata_of_db(lpath_db)

# # Dementia
# filter_criteria = {
#     "disease_type": ["AD", "DLB", "NPH", "HV"],
#     "cognitive_level": ["Normal", "Dementia", "MCI", "MCI23"],
#     # "institution": ["Osaka University"],  # "Osaka",
# }

# split_criteria = {
#     "subject_id":
#     "institution"
# } # to make datasets exclusive for subject and institution to validate the performance on unseen subject data aqcuired at unseen institution to avoid individual differences and site effects


# balancing_criteria = {
#     "disease_type":
# } # to balance the number of sumples between "AD", DLB, NPH, and HV


# meta_allow = apply_filter_criteria(meta_all, filter_criteria)
# ds = SQLiteDataset(
#     lpath_db=lpath_db,
#     filter_criteria=meta_allow,
#     show=False,
# )


import mngs
from scripts.data.database.SQLiteDataset import (
    SQLiteDataset,
    fetch_metadata_of_db,
)

# Initialize the configuration and global settings
CONFIG, sys.stdout, sys.stderr, plt, cc = mngs.gen.start(sys, plt)
WINDOW_SIZE_PTS = CONFIG["WINDOW_SIZE_SEC"] * CONFIG["TGT_FS"]

# Path to the SQLite database
lpath_db = "./data/Dementia/dbs/eeg_data.db"

# Fetch metadata from the database to understand the available data
meta_all = fetch_metadata_of_db(lpath_db)

# Define inclusion criteria to filter relevant data only
inclusion_criteria = {
    "disease_type": ["AD", "DLB", "NPH", "HV"],
    "cognitive_level": ["Normal", "Dementia", "MCI", "MCI23"],
    # Optionally include institutions if you're interested in specific sites
    "institution": [
        "Osaka University",
        "Kochi University",
        "Nippon Life Hospital",
    ],
}

# Create the dataset instance with the specified inclusion criteria
ds = SQLiteDataset(
    lpath_db=lpath_db, filter_criteria=inclusion_criteria, show=True
)

# Define split ratios and criteria for creating exclusive subsets
split_ratios = {"train": 0.6, "val": 0.2, "test": 0.2}
split_criteria = ["subject_id", "institution"]

# Split the dataset based on the specified criteria
subsets = ds.split_by_criteria(
    split_ratios=split_ratios, split_criteria=split_criteria
)

# Assuming 'disease_type' is the criterion for balancing the training dataset
balancing_criteria = "disease_type"

# Balance the training dataset
balanced_training_subset = ds.balance_training_dataset_by_criteria(
    subsets["train"], balancing_criteria
)

# Now, you can proceed to use these subsets for training, validation, and testing
# For example, to create DataLoader instances for each subset
train_loader = DataLoader(
    balanced_training_subset, batch_size=32, shuffle=True
)
val_loader = DataLoader(subsets["val"], batch_size=32, shuffle=False)
test_loader = DataLoader(subsets["test"], batch_size=32, shuffle=False)

# You can then use these DataLoader instances in your training loop
