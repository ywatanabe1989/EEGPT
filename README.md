python -u script/validate_bids.py | tee res/bids_validation.txt

channels.tsv

find ./data -name '*.tsv'

#### Dataset
- ds002778
[UC San Diego Resting State EEG Data from Patients with Parkinson's Disease](https://openneuro.org/datasets/ds002778/versions/1.0.5)
HC: 16
PD: 15

- ds003478
[EEG: Depression rest](https://openneuro.org/datasets/ds003478/versions/1.1.0/file-display/sub-002:eeg:sub-002_task-Rest_run-01_coordsystem.json)

- ds003775
[SRM Resting-state EEG](https://openneuro.org/datasets/ds003775/versions/1.2.1)

- ds004504
[A dataset of EEG recordings from: Alzheimer's disease, Frontotemporal dementia and Healthy subjects](https://openneuro.org/datasets/ds004504/versions/1.0.6)

# EDF_LPATHS = mngs.gen.find("./data", "*.edf")  # ds003775
# BDF_LPATHS = mngs.gen.find("./data", "*.bdf")  # ds002778
# FDT_LPATHS = mngs.gen.find("./data", "*.fdt")  # ds003478
# SET_LPATHS = mngs.gen.find("./data", "*.set")  # ds004504

# LPATHS = EDF_FILES + BDF_FILES + FDT_LPATHS + SET_LPATHS
