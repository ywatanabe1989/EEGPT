#!/bin/bash

download_an_openneuro_dataset() {
    ds_ID=$1
    dest_path="/mnt/ssd/EEGPT_datasets/OpenNeuro/${ds_ID}-download/"

    if [[ ! -d $dest_path ]]; then
        echo -e "\nDownloading ${ds_ID} ..."
        aws s3 sync --no-sign-request "s3://openneuro.org/${ds_ID}" "$dest_path" || echo -e "\n${ds_ID} download failed"
        echo -e "\n${ds_ID} download succeeded."
    else
        echo -e "\n${ds_ID} already exists. Skipping."
    fi
}

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <dataset_id>"
    exit 1
fi

download_an_openneuro_dataset "$1"

## EOF
