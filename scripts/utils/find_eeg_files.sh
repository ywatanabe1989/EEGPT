#!/bin/bash

function find_eeg_files() {
    ROOTDIR=$1
    SPATH=$2
    
    > $SPATH # [REVISED]
    EXTENSIONS=("eeg" "edf" "bdf" "gdf" "cnt" "egi" "mff" "set" "xdf" "xdfz")
    for ext in "${EXTENSIONS[@]}"; do
        find $ROOTDIR -name "*.$ext" >> $SPATH
    done
    echo -e "\n$(cat $SPATH | wc -l) files were found" # [REVISED]
    echo -e "\nSaved to: $SPATH"
}

find_eeg_files "./data/OpenNeuro/ds*/" "./data/OpenNeuro/eeg_files.txt" # [REVISED]
find_eeg_files "./data/Internal/ds*/" "./data/Internal/eeg_files.txt" # [REVISED]


# function find_eeg_files() {
#     ROOTDIR=$1
#     SPATH=$2
    
#     # Clear the file list before starting
#     > $SPATH
#     # Array of file extensions
#     EXTENSIONS=("eeg" "edf" "bdf" "gdf" "cnt" "egi" "mff" "set" "xdf" "xdfz")
#     # Loop through each file extension and append to file list
#     for ext in "${EXTENSIONS[@]}"; do
#         find $ROOTDIR -name "*.$ext" >> $SPATH
#     done
#     echo -e "\n`cat $SPATH | wc -l` files were found"
#     echo -e "\nSaved to: $SPATH"
# }

# find_eeg_files("./data/OpenNeuro/", "./data/OpenNeuro/eeg_files.txt")
# find_eeg_files("./data/Internal/", "./data/Internal/eeg_files.txt")



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

