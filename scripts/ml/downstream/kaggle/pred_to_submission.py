#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-03-11 15:18:58 (ywatanabe)"

# import mngs

# lpath = './scripts/ml/downstream/kaggle/clf_CNN1D/FINISHED/2024Y-03M-11D-14h14m13s_0B3c/submission.csv'
# submission = mngs.io.load(lpath)

#        eeg_id  seizure_vote  lpd_vote  gpd_vote  lrda_vote  grda_vote  other_vote
# 0  3911565283      0.277829  0.082979  0.077332   0.076036   0.155773    0.330052

columns = np.array(
    [
        "eeg_id",
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]
)
data = np.array(
    [3911565283, 0.277829, 0.082979, 0.077332, 0.076036, 0.155773, 0.330052]
)

sub_df = pd.DataFrame(pd.Series(index=columns, data=data)).T

sub_df.to_csv("submission.csv", index=False)
