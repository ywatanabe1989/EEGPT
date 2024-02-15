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
$ screen -dmS populate_MotorMovementImagery ./scripts/data/database/populate_db.py --data_category MotorMovementImagery
```


#### Updates links to databases
```
ls ./data/O*/dbs -alht

CAT=OpenNeuro
VER=v029
cd ./data/"$CAT"/dbs/ && ln -sf eeg_data_"$VER".db eeg_data.db && cd ../../..
```


#### Runs pretraining
``` bash
$ screen -dmS pretrain_beta_0.0 ./scripts/ml/pretrain_EEGPT.py -a 1.0 -b 0.0 -g 0.0
$ screen -dmS pretrain_beta_0.1 ./scripts/ml/pretrain_EEGPT.py -a 1.0 -b 0.1 -g 0.0
$ screen -dmS pretrain_beta_0.3 ./scripts/ml/pretrain_EEGPT.py -a 1.0 -b 0.3 -g 0.0
$ screen -dmS pretrain_beta_1.0 ./scripts/ml/pretrain_EEGPT.py -a 1.0 -b 1.0 -g 0.0
$ screen -dmS pretrain_beta_3.0 ./scripts/ml/pretrain_EEGPT.py -a 1.0 -b 3.0 -g 0.0
```




## OpenNeuro Datasets for Pretraining
Over 100 datasets identified on OpenNeuro under the following conditions:
- 1. MODALITY: EEG
- 2. PARTICIPANTS: 50 or more

From 78 datasets, EEG data of 6,144 subjects are utilized during pretraining.
