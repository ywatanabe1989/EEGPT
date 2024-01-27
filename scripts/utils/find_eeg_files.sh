#!/bin/bash

SPATH="./data/eeg_files.txt"

# Clear the file list before starting
> $SPATH

# Array of file extensions
EXTENSIONS=("eeg" "edf" "bdf" "gdf" "cnt" "egi" "mff" "set" "xdf" "xdfz")

# Loop through each file extension and append to file list
for ext in "${EXTENSIONS[@]}"; do
    find ./data/ -name "*.$ext" >> $SPATH
done

echo -e "\n`cat $SPATH | wc -l` files were found"
echo -e "\nSaved to: $SPATH"

# echo $unique_datasets
# BrainVision (.vhdr, .vmrk, .eeg)
# European data format (.edf)
# BioSemi data format (.bdf)
# General data format (.gdf)
# Neuroscan CNT (.cnt)
# EGI simple binary (.egi)
# EGI MFF (.mff)
# EEGLAB files (.set, .fdt)
# Nihon Kohden EEG data (.eeg, .21e, .pnt, .log)
# XDF data (.xdf, .xdfz)

