<details>
<summary>Installation</summary>

```
$ git clone git@github.com:ywatanabe1989/EEGPT.git && cd EEGPT
$ echo 'export PYTHONPATH=.:$PYTHONPATH' >> ~/.bash_profile # add current directory to PYTHONPATH
```

</details>


## Scripts

<details>
<summary>Dataset Handling</summary>

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

</details>

<details>
<summary>Pretraining</summary>

``` bash
./scripts/ml/pretrain/EEGPT.py -h
# screen -dmS "$session_name" bash -c "$command"
# ./scripts/ml/pretrain/EEGPT/FINISHED/2024Y-03M-01D-03h26m30s_EJgv/

screen -dmS kaggle ./scripts/ml/downstream/kaggle/clf_CNN1D.py
screen -dmS kaggle ./scripts/ml/downstream/epilepsy/clf_CNN1D.py
./scripts/ml/downstream/epilepsy/visualize.py
```

</details>

<details>
<summary>Hyper parameters</summary>

[`./scripts/ml/parse_model_configs.py`](./scripts/ml/parse_model_configs.py) handles hyper parameters.

``` bash
./scripts/ml/parse_model_configs.py -h | tee ./docs/hyper_parameters.txt
```

</details>

<details>
<summary>TODO</summary>

## TODO
- [ ] Shuffle segments based on time axis
- [ ] Handle low-freq bands considering segment length
- [ ] Visualize UMAP embedding by taking average for each subject

</details>


<details>
<summary>EEG features</summary>

- [ ] FFT amplitude
- [ ] FFT phase
- [ ] Hilbert
- [ ] PAC
- [ ] spectrum
- [ ] ACT
- [ ] Information Flow
  Transfer Entropy or Directed Transfer Function
- [ ] Graph
  coherence or phase lag index
  Analyze the graph using metrics like clustering coefficient, path length, or centrality to study brain connectivity patterns.
- [ ] CNN
- [ ] DTW
- [ ] log1p

</details>

Currently, I am trying to create a universal EEG model.

EEG signals from over 7,000 subjects have already been collected.

I have confirmed that the FFT band amplitude of 1-second EEG data can be predicted from the preceding 8 seconds of EEG data.

I believe that experimenting with multiple features and switching them with Optuna will be beneficial for exploring the potential of future prediction tasks as a pretraining for EEG.

Also, as a objective value, I would like to use silhouette score on the latent features for dementia subtypes.

from 1 seconds of 14-channel EEG data sampled at 128 Hz, what kinds of features would be interesting?

1 second of EEG data:

<details>
<summary>Plots</summary>

Problem Definition

#### Hypotheses
Hypotheses	Summary
- High-quality representations inherently possess discriminative power for different conditions.
  
- A diverse dataset collection enhances the model's ability to generalize across various settings.
  
- Methodical tuning of hyperparameters significantly improves the quality of feature representations.
  
- Pretraining models in an unsupervised manner on diverse datasets boosts performance on specific downstream tasks.
  
- The discriminability and generalizability of latent features can indicate a model’s effectiveness in transfer learning.
  
- Training with a wide range of data sources mitigates bias, leading to fairer and more equitable models.


#### Hypothesis 2


</details>
