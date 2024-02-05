#!/bin/bash

# Define the source and destination directories
SOURCE_DIR="./scripts/"
DEST_DIR="./tests/"

# Check if the source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Source directory does not exist: $SOURCE_DIR"
    exit 1
fi

# Create the destination directory if it does not exist
if [ ! -d "$DEST_DIR" ]; then
    mkdir -p "$DEST_DIR"
fi

# Use find to iterate over all files in the source directory
find "$SOURCE_DIR" -type f -name "*.py" -exec bash -c ' \
    SOURCE_DIR="$0"; \
    DEST_DIR="$1"; \
    FILE_PATH="$2"; \
    FILE_NAME="${FILE_PATH##*/}"; \
    # Check if the file already has an underscore prefix
    if [[ "$FILE_NAME" == _* ]]; then
        DEST_FILE_NAME="$FILE_NAME"
    else
        DEST_FILE_NAME="_$FILE_NAME"
    fi
    DEST_SUBDIR="${FILE_PATH#$SOURCE_DIR}"; \
    DEST_SUBDIR="${DEST_SUBDIR%$FILE_NAME}"; \
    DEST_PATH="$DEST_DIR$DEST_SUBDIR"; \
    mkdir -p -- "$DEST_PATH"; \
    # Copy the file, using the determined destination file name
    cp -- "$FILE_PATH" "$DEST_PATH/$DEST_FILE_NAME"' "$SOURCE_DIR" "$DEST_DIR" {} \;

# Inform the user that the process is complete
echo "Mirroring complete with underscore prefix for files."

