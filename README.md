## Installation
```
$ git clone git@github.com:ywatanabe1989/EEGPT.git
$ cd EEGPT
$ python3 -m venv env
$ source env/bin/activate
$ pip install -U pip
$ pip install -Ur requirements.txt
echo 'export PYTHONPATH=.:$PYTHONPATH' >> ~/.bash_profile
```

## Scripts
```
# edit ./config/is_debug.yaml
$ ./scripts/utils/download_OpenNeuro_datasets.sh # Takes a large amount of time
$ yes | ./scripts/utils/populate_db.py
$ ./scripts/utils/summarize_datasets_as_a_csv.py 

$ yes | ./scripts/utils/populate_db.py && ./scripts/train_EEGPT.py
```

## OpenNeuro Datasets for Pretraining
47 datasets identified on OpenNeuro under the following conditions:
- 1. MODALITY: EEG
- 2. PARTICIPANTS: 50 or more

``` bash
$ du data/OpenNeuro/ds* | wc -l # 47
$ du data/OpenNeuro # 3.2T
$ du data/OpenNeuro/ds*
# 3.1G    data/OpenNeuro/ds003474-download
# 3.1G    data/OpenNeuro/ds003478-download
# 3.1G    data/OpenNeuro/ds003490-download
# 3.1G    data/OpenNeuro/ds003506-download
# 3.1G    data/OpenNeuro/ds003509-download
# 3.1G    data/OpenNeuro/ds003518-download
# 3.1G    data/OpenNeuro/ds003522-download
# 3.1G    data/OpenNeuro/ds003523-download
# 3.1G    data/OpenNeuro/ds003602-download
# 3.1G    data/OpenNeuro/ds003638-download
# 56G     data/OpenNeuro/ds003825-download
# 254G    data/OpenNeuro/ds003838-download
# 6.2G    data/OpenNeuro/ds003944-download
# 13G     data/OpenNeuro/ds003947-download
# 55G     data/OpenNeuro/ds003969-download
# 18G     data/OpenNeuro/ds004019-download
# 101G    data/OpenNeuro/ds004067-download
# 125G    data/OpenNeuro/ds004118-download
# 303G    data/OpenNeuro/ds004120-download
# 95G     data/OpenNeuro/ds004148-download
# 23G     data/OpenNeuro/ds004151-download
# 78G     data/OpenNeuro/ds004166-download
# 215G    data/OpenNeuro/ds004186-download
# 19G     data/OpenNeuro/ds004256-download
# 26G     data/OpenNeuro/ds004279-download
# 9.9G    data/OpenNeuro/ds004315-download
# 19G     data/OpenNeuro/ds004317-download
# 12G     data/OpenNeuro/ds004362-download
# 1.2T    data/OpenNeuro/ds004395-download
# 36G     data/OpenNeuro/ds004448-download
# 5.4G    data/OpenNeuro/ds004504-download
# 9.5G    data/OpenNeuro/ds004515-download
# 23G     data/OpenNeuro/ds004532-download
# 44G     data/OpenNeuro/ds004572-download
# 14G     data/OpenNeuro/ds004574-download
# 660M    data/OpenNeuro/ds004577-download
# 25G     data/OpenNeuro/ds004579-download
# 16G     data/OpenNeuro/ds004580-download
# 2.9G    data/OpenNeuro/ds004584-download
# 7.9G    data/OpenNeuro/ds004595-download
# 74G     data/OpenNeuro/ds004602-download
# 20G     data/OpenNeuro/ds004626-download
# 30G     data/OpenNeuro/ds004635-download
# 102G    data/OpenNeuro/ds004718-download
# 1.4G    data/OpenNeuro/ds004771-download
# 117G    data/OpenNeuro/ds004883-download
# 8.1G    data/OpenNeuro/ds004902-download
```

## Epilepsy Datasets

``` bash
$ du data/Dementia/ds* | wc -l # 3
$ du data/Dementia # 53G
$ du data/Dementia/*
11G     data/Dementia/BIDS_dataset_Nissei_v1.1
37G     data/Dementia/BIDS_dataset_Osk_v5.3
5.8G    data/Dementia/BIDS_dataset_v1.1_Kochi
```

## WorkingMemory Datasets
``` bash
$ du data/WorkingMemory/ds* | wc -l # 1
$ du data/WorkingMemory # 38G
$ du data/WorkingMemory/*
38G     data/WorkingMemory/ds004395-download
```

## MotorMovementImagery
``` bash
$ du data/MotorMovementImagery/* | wc -l # 1
$ du data/MotorMovementImagery/*
3.4G    data/MotorMovementImagery/eeg-motor-movementimagery-dataset-1.0.0
```


