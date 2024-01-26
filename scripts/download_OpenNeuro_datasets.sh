#!/bin/bash

# OpenNeuro
# EEG
# >= 50
# https://openneuro.org/search/modality/eeg?query=%7B%22modality_selected%22%3A%22EEG%22%2C%22subjectCountRange%22%3A%5B50%2Cnull%5D%7D

# Declare ds_list as a bash array
ds_ID_list=(
    ds004902
    ds004883
    ds004771
    ds004718
    ds004635
    ds004626
    ds004602
    ds004595
    ds004584
    ds004580
    ds004579
    ds004577
    ds004574
    ds004572
    ds004532
    ds004515
    ds004504
    ds004448
    ds004395
    ds004362
    ds004317
    ds004315
    ds004279
    ds004256
    ds004186
    ds004166
    ds004151
    ds004148
    ds004120
    ds004118
    ds004067
    ds004019
    ds003969
    ds003947
    ds003944
    ds003838
    ds003825
    ds003775
    ds003690
    ds003655
    ds003638
    ds003602
    ds003523
    ds003522
    ds003518
    ds003509
    ds003506
    ds003490
    ds003478
    ds003474    
)

find /mnt/ssd/EEGPT_datasets/ -type d -empty -delete
echo "${ds_ID_list[@]}" | tr ' ' '\n' | xargs -I {} -n1 -P10 ./scripts/download_an_OpenNeuro_dataset.sh {}
find /mnt/ssd/EEGPT_datasets/ -type d -empty -delete

# rm -rf /mnt/ssd/EEGPT_datasets/ds*-download
# screen -S download
# ./scripts/download_OpenNeuro_datasets.sh | tee ./download_OpenNeuro_datasets.log
## EOF
