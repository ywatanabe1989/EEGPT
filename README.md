## Installation
```
$ git clone git@github.com:ywatanabe1989/EEGPT.git
$ cd EEGPT
$ python -m venv env
$ source env/bin/activate
$ pip install -U pip
$ pip install -r requirements.txt
```

## Scripts
```
$ source env/bin/activate
$ ./scripts/download_OpenNeuro_datasets.sh
$ ./scripts/utils/populate_db.py
$ ./scripts/train_EEGPT.py
$ ./scripts/plot_learning_curves.py
```

