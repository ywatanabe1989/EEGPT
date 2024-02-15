#!/bin/bash

# Define an associative array with category names as keys and paths as values
declare -A categories=(
    [OpenNeuro]="data/OpenNeuro/raw"
    [Dementia]="data/Dementia/raw"
    [Epilepsy]="data/Epilepsy/raw"
    [WorkingMemory]="data/WorkingMemory/raw"
    [MotorMovementImagery]="data/MotorMovementImagery/raw"
)

# Function to summarize disk usage for a given dataset
print_dataset() {
    local dataset_path=$1
    local dataset_name=$2

    echo "## $dataset_name Datasets"
    echo
    echo "\$ du $dataset_path # $(du -sh $dataset_path | cut -f1)"
    echo "\$ du $dataset_path/ds* | wc -l # $(find $dataset_path -maxdepth 1 -type d -name 'ds*' | wc -l)"
    echo "\$ du $dataset_path/ds*"
    du -sh $dataset_path/ds* | sort -hr
    echo
}


# Loop through the categories and summarize disk usage for each dataset
for category in "${!categories[@]}"; do
    print_dataset "${categories[$category]}" "$category"
done

# EOF
