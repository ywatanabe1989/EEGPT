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
$ screen -dmS populate_Kaggle ./scripts/ml/downstream/kaggle/populate_db.py
```


#### Updates links to databases
```
CAT=OpenNeuro
CAT=Dementia
CAT=Kaggle
CAT=Epilepsy

ls ./data/"$CAT"/dbs -alht
VER=v010

function ln_new_db() {
    CAT=$1
    VER=$2
    cd ./data/"$CAT"/dbs/ && ln -sf eeg_data_"$VER".db eeg_data.db && cd ../../..
    rm -f ./data/"$CAT"/dbs/eeg_data_meta_cache.pkl
    ls ./data/"$CAT"/dbs -alht
    }
    
ln_new_db $CAT $VER
```


#### Runs pretraining
``` bash
./scripts/ml/pretrain/EEGPT.py -h
# screen -dmS "$session_name" bash -c "$command"
# ./scripts/ml/pretrain/EEGPT/FINISHED/2024Y-03M-01D-03h26m30s_EJgv/

screen -dmS kaggle ./scripts/ml/downstream/kaggle/clf_CNN1D.py
screen -dmS kaggle ./scripts/ml/downstream/epilepsy/clf_CNN1D.py
./scripts/ml/downstream/epilepsy/visualize.py
```


## Hyper-parameters to Tune
[`./scripts/ml/parse_model_configs.py`](./scripts/ml/parse_model_configs.py) handles hyper parameters.

``` bash
./scripts/ml/parse_model_configs.py -h | tee ./docs/hyper_parameters.txt
```

