#!/bin/bash

# Define the number of minutes to check for last modification
# Set this to a value that makes sense for your downloads
# For example, if a dataset hasn't been modified in the last 10 minutes, it might be complete
minutes_since_last_update=30

# Change to the directory where datasets are being downloaded
# cd "$source_dir"

# Find directories that have not been modified in the last specified number of minutes
# and move them to the 'raw' directory
find . -maxdepth 1 -type d -name 'ds*-download' -mmin +$minutes_since_last_update -exec sh -c '
    for dataset do
        rsync -av --update "$dataset"/ ./raw/
        echo "Synced $dataset to ./raw"
    done
' sh {} +

# find . -maxdepth 1 -type d -name 'ds*-download' -mmin +$minutes_since_last_update -exec sh -c '
#     for dataset do
#         mv "$dataset" ./raw
#         echo "Moved $dataset to ./raw
#     done
# ' sh {} +

# EOF
