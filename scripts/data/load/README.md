## [HMS - Harmful Brain Activity Classification](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/overview)
Classify seizures and other patterns of harmful brain activity in critically ill patients

| Pattern | Description                             |
|---------|-----------------------------------------|
| SZ      | Seizure                                 |
| GPD     | Generalized Periodic Discharges         |
| LPD     | Lateralized Periodic Discharges         |
| LRDA    | Lateralized Rhythmic Delta Activity     |
| GRDA    | Generalized Rhythmic Delta Activity     |
| Other   | Other EEG patterns not classified above |


| File Group              | Description                                                                                                                                                                                                                                                                                                  |
|-------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `train.csv`             | Metadata for the train set with labels for EEG samples and spectrograms. Includes `eeg_id`, `eeg_sub_id`, `eeg_label_offset_seconds`, `spectrogram_id`, `spectrogram_sub_id`, `spectogram_label_offset_seconds`, `label_id`, `patient_id`, `expert_consensus`, and `[seizure/lpd/gpd/lrda/grda/other]_vote`. |
| `test.csv`              | Metadata for the test set with no overlapping samples. Includes `eeg_id`, `spectrogram_id`, `patient_id`.                                                                                                                                                                                                    |
| `sample_submission.csv` | Template for submissions with `eeg_id` and predicted probabilities for `[seizure/lpd/gpd/lrda/grda/other]_vote`.                                                                                                                                                                                             |
| `train_eegs/`           | EEG data from overlapping samples. Metadata in `train.csv` is used to select subsets. Columns are electrode locations and EKG data.                                                                                                                                                                          |
| `test_eegs/`            | Exactly 50 seconds of EEG data for the test set.                                                                                                                                                                                                                                                             |
| `train_spectrograms/`   | Spectrograms from EEG data. Metadata in `train.csv` is used to select subsets. Columns indicate frequency and recording regions.                                                                                                                                                                             |
| `test_spectrograms/`    | Spectrograms from exactly 10 minutes of EEG data for the test set.                                                                                                                                                                                                                                           |
| `example_figures/`      | Larger copies of example images used on the overview tab.                                                                                                                                                                                                                                                    |

