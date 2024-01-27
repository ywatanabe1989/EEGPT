## Installation
```
$ git clone git@github.com:ywatanabe1989/EEGPT.git

# Python

$ cd EEGPT
$ python3 -m venv env
$ source env/bin/activate
$ pip install -U pip
$ pip install -r requirements.txt
```

## Scripts
```
$ ./scripts/utils/download_OpenNeuro_datasets.sh
$ ./scripts/utils/populate_db.py
$ ./scripts/utils/summarie_datasets_as_a_csv.py 
$ ./scripts/train_EEGPT.py
```

## Datasets
49 datasets identified on OpenNeuro under the following conditions:
- 1. MODALITY: EEG
- 2. PARTICIPANTS: 50 or more

``` bash
$ du data/ds* | wc -l # 38
$ du data/ # 1.7T
$ du data/ds*
# 44G     data/ds003825-download
# 54G     data/ds003838-download
# 6.2G    data/ds003944-download
# 13G     data/ds003947-download
# 55G     data/ds003969-download
# 18G     data/ds004019-download
# 94G     data/ds004067-download
# 96G     data/ds004118-download
# 100G    data/ds004120-download
# 95G     data/ds004148-download
# 23G     data/ds004151-download
# 78G     data/ds004166-download
# 114G    data/ds004186-download
# 19G     data/ds004256-download
# 26G     data/ds004279-download
# 9.9G    data/ds004315-download
# 19G     data/ds004317-download
# 12G     data/ds004362-download
# 146G    data/ds004395-download
# 36G     data/ds004448-download
# 5.4G    data/ds004504-download
# 9.5G    data/ds004515-download
# 23G     data/ds004532-download
# 4.8G    data/ds004572-download
# 14G     data/ds004574-download
# 660M    data/ds004577-download
# 25G     data/ds004579-download
# 16G     data/ds004580-download
# 2.9G    data/ds004584-download
# 7.9G    data/ds004595-download
# 74G     data/ds004602-download
# 20G     data/ds004626-download
# 30G     data/ds004635-download
# 102G    data/ds004718-download
# 1.4G    data/ds004771-download
# 117G    data/ds004883-download
# 8.1G    data/ds004902-download

```

## Populate dataset to SQLite3 data base

``` bash
$ ./scripts/utils/populate_db.py # ./data/eeg_data_v7.db # /2024-0127-12:41_yPiZGl7l/
```


## [MNE documentation](https://mne.tools/stable/auto_tutorials/io/20_reading_eeg_data.html)
- BrainVision (.vhdr, .vmrk, .eeg)
- European data format (.edf)
- BioSemi data format (.bdf)
- General data format (.gdf)
- Neuroscan CNT (.cnt)
- EGI simple binary (.egi)
- EGI MFF (.mff)
- EEGLAB files (.set, .fdt)
- Nihon Kohden EEG data (.eeg, .21e, .pnt, .log)
- XDF data (.xdf, .xdfz)


