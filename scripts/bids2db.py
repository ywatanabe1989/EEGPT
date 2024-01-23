#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-01-23 15:35:50 (ywatanabe)"

import sqlite3
from torch.utils.data import Dataset, DataLoader
import mne

def get_10_20_montage():
    montage = mne.channels.make_standard_montage("standard_1020")    
    montage.rename_channels(
        lambda x: x.upper()
    )  # Convert channel names to uppercase
    # self.montage = montage
    return montage
    

# Assuming you have a function that preprocesses a single file and returns a segment
def preprocess_eeg_file(filepath):
    channel_mapping = {
        "T3": "T7",
        "T4": "T8",
        "T5": "P7",
        "T6": "P8",
    }
    raw = mne.io.read_raw(filepath, preload=True, verbose=False)
    raw.rename_channels(
        lambda x: x.upper()
    )  # Convert channel names to uppercase
    channel_mapping = {
        old: new
        for old, new in channel_mapping.items()
        if old in raw.ch_names
    }
    raw.rename_channels(channel_mapping)
    raw.set_montage(self.montage, on_missing="ignore")
    # Pick the channels that are present in both the raw data and the montage
    raw = raw.pick(self.common_channels)
    raw.resample(self.sampling_rate)

    # Randomly select a time segment
    total_time = raw.times[-1]
    start_time = random.uniform(0, total_time - self.t_segment)
    stop_time = start_time + self.t_segment

    # Extract EEG data for the time segment
    segment, times = raw[
        :,
        int(start_time * raw.info["sfreq"]) : int(
            stop_time * raw.info["sfreq"]
        ),
    ]

    # Common average
    segment -= segment.mean(axis=0, keepdims=True)

    # Normalize the data
    segment = (segment - np.mean(segment)) / np.std(segment)

    return torch.from_numpy(np.array(segment)).float()
    
    # # ... your preprocessing code ...
    # return processed_segment

# Create and populate the database
def create_and_populate_db(files_list, db_path='eeg_data.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS eeg_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        segment BLOB NOT NULL
    )
    ''')
    for filepath in files_list:
        segment = preprocess_eeg_file(filepath)
        cursor.execute('INSERT INTO eeg_data (segment) VALUES (?)', (segment,))
    conn.commit()
    conn.close()

# Modify the BidsDataset to load from the database
class BidsDatasetSQLite(Dataset):
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute('SELECT COUNT(*) FROM eeg_data')
        self.length = self.cursor.fetchone()[0]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        self.cursor.execute('SELECT segment FROM eeg_data WHERE id = ?', (index+1,))
        segment = self.cursor.fetchone()[0]
        return torch.from_numpy(np.frombuffer(segment, dtype=np.float32))

    def close(self):
        self.conn.close()

if __name__ == "__main__":
    # Usage
    import mngs
    FILES_LIST = mngs.io.load("files_list_tmp.txt") # bdf
    create_and_populate_db(FILES_LIST[:3])

    
    db_path = 'eeg_data.db'
    bids_dataset_sqlite = BidsDatasetSQLite(db_path=db_path)
    data_loader = DataLoader(bids_dataset_sqlite, batch_size=batch_size, shuffle=True, num_workers=4)
     
    # Don't forget to close the database connection when done
    bids_dataset_sqlite.close()


