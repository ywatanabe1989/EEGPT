## Installation
```
$ git clone git@github.com:ywatanabe1989/EEGPT.git && cd EEGPT
$ echo 'export PYTHONPATH=.:$PYTHONPATH' >> ~/.bash_profile # add current directory to PYTHONPATH
```

## Scripts
#### Toggles the debugging state
```
$ ./toggle_debug_status.sh # or manually edit the './config/IS_DEBUG.yaml' file
```

#### Downloads OpenNeuro datasets
``` bash
$ ./scripts/data/download/download_OpenNeuro_datasets.sh # Takes a significant amount of time
$ ./scripts/utils/summarize_datasets_as_a_csv.py 
$ ./scripts/utils/print_datasets.sh | tee ./docs/datasets.txt
```

#### Populates BIDS datasets into SQLite3 databases

``` bash
$ screen -dmS populate_OpenNeuro ./scripts/data/database/populate_db.py --data_category OpenNeuro
$ screen -dmS populate_Dementia ./scripts/data/database/populate_db.py --data_category Dementia
$ screen -dmS populate_Epilepsy ./scripts/data/database/populate_db.py --data_category Epilepsy
$ screen -dmS populate_MotorMentalMovement ./scripts/data/database/populate_db.py --data_category MotorMentalMovement
```


#### Updates the links to databases
```
ls ./data/*/dbs -alht

cd ./data/OpenNeuro/dbs
ln -s eeg_data_v024.db eeg_data.db --force
cd ../../..

cd ./data/Dementia/dbs
ln -s eeg_data_v011.db eeg_data.db --force
cd ../../..

cd ./data/Epilepsy/dbs
ln -s eeg_data_v004.db eeg_data.db --force
cd ../../..

cd ./data/MotorMovementImagery/dbs
ln -s eeg_data_v001.db eeg_data.db --force
cd ../../..
```


#### Runs pretraining
``` bash
$ ./scripts/ml/pretrain_EEGPT.py
```




## OpenNeuro Datasets for Pretraining
47 datasets identified on OpenNeuro under the following conditions:
- 1. MODALITY: EEG
- 2. PARTICIPANTS: 50 or more

```
Congratulations! SQLite database was created from 41823 files, 19 datasets, and 4450 subjects:
./data/OpenNeuro/dbs/eeg_data_v016.db
```


