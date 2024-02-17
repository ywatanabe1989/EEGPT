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
VER=v030
cd ./data/"$CAT"/dbs/ && ln -sf eeg_data_"$VER".db eeg_data.db && cd ../../..
```


#### Runs pretraining
``` bash
# 1st Pretrain: the Convolution and VectorQuantization Layers
screen -dmS pretrain_1st ./scripts/ml/pretrain_EEGPT.py --commit_coef 1.0 --entropy_coef 0.3 --pred_coef 0.0 --softmax_type gumbel_softmax_hard

# 2nd Pretrain: the Transformer Layers
screen -dmS pretrain_2nd ./scripts/ml/pretrain_EEGPT.py --commit_coef 0. --entropy_coef 0. --pred_coef 1. --softmax_type gumbel_softmax_hard --conv_vq_weights_path ./scripts/ml/pretrain_EEGPT/2024-02-17-01-48_KTyjYpl1/checkpoint.pth

screen -dmS pretrain_1st ./scripts/ml/pretrain_EEGPT.py --commit_coef 1.0 --entropy_coef 0.3 --pred_coef 0.0 --softmax_type gumbel_softmax_hard --num_embeddings 16

# 2024-02-17-08-15_UmciKUpN
# 2024-02-17-08-19_0qqNiw9a
```




## OpenNeuro Datasets for Pretraining
Over 100 datasets identified on OpenNeuro under the following conditions:
- 1. MODALITY: EEG
- 2. PARTICIPANTS: 50 or more

From 78 datasets, EEG data of 6,144 subjects are utilized during pretraining.

## Hyper-parameters to Tune
```
LEARNING_RATE
BATCH_SIZE
OPTIMIZER
WINDOW_SIZE_SEC
N_EPOCHS
DROPOUT_RATIO

VQ_NUM_EMBEDDINGS
VQ_EMBEDDING_DIM
VQ_SOFTMAX_TYPE
VQ_LOSS_COEFFICIENTS
TRANSFORMER_N_HEADS
TRANSFORMER_HIDDEN_DIM
TRANSFORMER_N_ENC_LAYERS
TRANSFORMER_N_DEC_LAYERS
```
