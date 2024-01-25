#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-01-23 12:56:17 (ywatanabe)"

import sqlite3
import random

# Create a new SQLite database
db_path = 'demo.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create a table
cursor.execute('''
CREATE TABLE IF NOT EXISTS data_table (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    features INTEGER NOT NULL,
    label INTEGER NOT NULL
)
''')

# Insert some sample data
for i in range(100):  # Let's insert 100 records
    features = random.randint(0, 100)  # Random integer feature
    label = features % 2  # Simple binary label based on even/odd
    cursor.execute('INSERT INTO data_table (features, label) VALUES (?, ?)', (features, label))

# Commit changes and close the connection
conn.commit()
conn.close()

print("Database and sample data created successfully.")
