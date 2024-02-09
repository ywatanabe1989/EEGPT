#!/bin/bash

# Path to the file containing dataset IDs
DATASET_IDS_FILE="./scripts/data/download/OpenNeuro_EEG_datasts_IDs.txt"

# Path to the download script
DOWNLOAD_SCRIPT="./scripts/data/download/_download_an_OpenNeuro_dataset.sh"

# Delete any empty directories in ./data/ before starting the download
find ./data/ -type d -empty -delete

cat $DATASET_IDS_FILE | xargs -I {} -n1 -P10 $DOWNLOAD_SCRIPT {}

find ./data/ -type d -empty -delete

# EOF
