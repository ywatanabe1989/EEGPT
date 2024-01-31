#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-01-31 10:39:25 (ywatanabe)"

import argparse

import pandas as pd


def excel2csv(excel_file_path, csv_file_path):
    # Load the Excel file
    df = pd.read_excel(excel_file_path)

    # Save the dataframe to a CSV file
    df.to_csv(csv_file_path, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert Excel to CSV")
    parser.add_argument("lpath", type=str, help="Path to the Excel file")
    args = parser.parse_args()  # [REVISED]
    excel2csv(args.lpath, args.lpath.replace(".xlsx", ".csv"))
