#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-06 17:49:11 (ywatanabe)"


def check_db_with_pandas(lpath_db):
    import sqlite3

    import pandas as pd

    # Connect to the SQLite database
    conn = sqlite3.connect(lpath_db)

    # Read all data from the table into a pandas DataFrame
    df = pd.read_sql_query("SELECT * FROM eeg_data", conn)

    # Close the connection to the database
    conn.close()
    return df


if __name__ == "__main__":
    lpath_db = "./data/Dementia/dbs/checked/eeg_data_v005.db"
    df = check_db_with_pandas(lpath_db)
    print(df)
