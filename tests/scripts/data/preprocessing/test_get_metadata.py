#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-06 08:25:25 (ywatanabe)"

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import mngs


def test_get_metadata():
    # Load configurations
    CONFIG = mngs.io.load_configs(show=False)

    # Import the get_metadata function
    from scripts.data.preprocessing.get_metadata import get_metadata

    # Define a helper function to print metadata
    def print_metadata(file_path):
        try:
            metadata = get_metadata(file_path)
            print(metadata)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    # Test cases for different datasets
    # OpenNeuro
    print_metadata(
        "./data/OpenNeuro/raw/ds003474-download/sub-001/eeg/sub-001_task-ProbabilisticSelection_eeg.fdt"
    )

    # Dementia, Japan
    print_metadata(
        "./data/Dementia/raw/BIDS_dataset_Osk_v5.3/sub-EEGO0001/ses-010/eeg/sub-EEGO0001_ses-010_task-rest_run-01_eeg.edf"
    )

    # Epilepsy, Japan
    # FIXME: Add test case for Epilepsy, Japan when ready

    # Epilepsy, Siena
    lpaths_siena = mngs.gen.find(
        "./data/Epilepsy/raw/siena-scalp-eeg-database-1.0.0/", "f", "*.edf"
    )
    for lpath_siena in lpaths_siena:
        print_metadata(lpath_siena)

    # MotorMovementImagery
    # FIXME: Segmentation should be applied beforehand
    print_metadata(
        "./data/MotorMovementImagery/raw/eeg-motor-movementimagery-dataset-1.0.0/S001/S001R03.edf"
    )


if __name__ == "__main__":
    test_get_metadata()
