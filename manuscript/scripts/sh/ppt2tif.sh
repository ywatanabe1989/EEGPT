#!/bin/bash

inputFilePath=$1

echo -e "\nConverting ${inputFilePath}..."

inputFilePathWin=$(wslpath -w "$inputFilePath")

if [ "$#" -eq 2 ]; then
    outputFilePath=$2
    outputFilePathWin=$(wslpath -w "$outputFilePath")
else
    outputFilePathWin=${inputFilePathWin%.pptx}.tif
fi

try:
# Run the PowerShell script
powershell.exe -ExecutionPolicy Bypass -File "$(wslpath -w ./scripts/ps1/ppt2tif.ps1)" -inputFilePath "$inputFilePathWin" -outputFilePath "$outputFilePathWin"

# if [ -n "$inputFilePath" ]; then
#    echo -e "\nConverted: ${inputFilePath} -> ${inputFilePath%.pptx}.tif"
# fi

if [ $? -eq 0 ]; then
    echo -e "\nConversion successful: ${inputFilePath} -> ${outputFilePath}"
else
    echo -e "\nConversion failed for: ${inputFilePath}"
fi

# ./scripts/sh/ppt2tif.sh src/figures/Figure_ID_01.pptx

## EOF
