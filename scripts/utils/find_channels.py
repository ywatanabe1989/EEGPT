#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-01-27 09:20:40 (ywatanabe)"


import sys

import matplotlib.pyplot as plt
import mne
import mngs
import numpy as np
import pandas as pd
from scipy.spatial import distance


# Functinos
def cvt_to_1020_chs(raw, tgt_montages):
    chs_df = pd.DataFrame(raw.get_montage().get_positions()["ch_pos"]).T
    # Set the standard 10-20 montage
    STANDARD_MONTAGE = mne.channels.make_standard_montage("standard_1020")
    STANDARD_DF = pd.DataFrame(STANDARD_MONTAGE.get_positions()["ch_pos"]).T

    link_dict = {}
    for tgt_montage in tgt_montages:
        tgt_coord = STANDARD_DF.T[tgt_montage.capitalize()]
        distances = ((chs_df - np.array(tgt_coord)[np.newaxis]) ** 2).sum(
            axis=1
        )
        # plt.hist(distances)
        min_dist = distances.min()
        print(min_dist)
        i_min = np.where(distances == min_dist)[0][0]
        closest_ch = chs_df.iloc[i_min].name
        link_dict[closest_ch] = tgt_montage

    # set found closest montages
    # raw.................. .fixme
    return raww
    # return link_dict


def my_load_raw(lpath):
    montages_upper = [ch.upper() for ch in raw.get_montage().ch_names]
    is_montage_ok = len(
        mngs.gen.search(
            [f"^{tm}" for tm in CONFIG["TGT_MONTAGE"]], montages_upper
        )[1]
    ) == len(CONFIG["TGT_MONTAGE"])
    if is_montage_ok:
        return raw


# Configurations
CONFIG, sys.stdout, sys.stderr, plt = mngs.gen.start(
    sys,
    plt,
    show=False,
)

# Load path
# # g06
# lpath = "./data/ds004635-download/sub-S9/eeg/sub-S9_task-resting_eeg.set"
# lpath = "./data/ds003775-download/sub-106/ses-t2/eeg/sub-106_ses-t2_task-resteyesc_eeg.edf"
# lpath = "./data/ds003969-download/sub-098/eeg/sub-098_task-think2_eeg.bdf"

lpath = "./data/ds004902-download/sub-40/ses-1/eeg/sub-40_ses-1_task-eyesopen_eeg.set"

raw = mne.io.read_raw(lpath)


raw.set_montage(
    mne.channels.make_standard_montage("standard_1020")
)  # [REVISED]
raw = cvt_to_1020_chs(raw)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-01-26 20:58:43 (ywatanabe)"


# import sys

# import matplotlib.pyplot as plt
# import mne
# import mngs
# import numpy as np
# import pandas as pd
# from scipy.spatial import distance


# # Functinos
# def cvt_to_1020_chs(raw):
#     chs_df = pd.DataFrame(raw.get_montage().get_positions()["ch_pos"]).T
#     # Set the standard 10-20 montage
#     STANDARD_MONTAGE = mne.channels.make_standard_montage("standard_1020")
#     STANDARD_DF = pd.DataFrame(STANDARD_MONTAGE.get_positions()["ch_pos"]).T

#     link_dict = {}
#     for tgt_montage in CONFIG["TGT_MONTAGE"]:
#         tgt_coord = STANDARD_DF.T[tgt_montage.capitalize()]
#         distances = ((chs_df - np.array(tgt_coord)[np.newaxis]) ** 2).sum(
#             axis=1
#         )
#         min_dist = distances.min()
#         print(min_dist)
#         i_min = np.where(distances == min_dist)[0][0]
#         closest_ch = chs_df.iloc[i_min].name
#         link_dict[closest_ch] = tgt_montage

#     return raw
#     # return link_dict


# # Configurations
# CONFIG, sys.stdout, sys.stderr, plt = mngs.gen.start(
#     sys,
#     plt,
#     show=False,
# )

# # Load path
# lpath = "./data/ds004635-download/sub-S9/eeg/sub-S9_task-resting_eeg.set"
# # lpath = "./data/ds003775-download/sub-106/ses-t2/eeg/sub-106_ses-t2_task-resteyesc_eeg.edf"
# # lpath = "./data/ds003969-download/sub-098/eeg/sub-098_task-think2_eeg.bdf"


# # Main
# raw = mne.io.read_raw(lpath)
# raw.set_montage(mne.channels.make_standard_montage("standard_1020"))  # [REVISED]
# raw = cvt_to_1020_chs(raw, CONFIG["TGT_MONTAGE"])
# )
# # raw = mne.io.read_raw(lpath)

# # {'E25': 'FP1',
# #  'E24': 'F3',
# #  'E42': 'C3',
# #  'E60': 'P3',
# #  'E70': 'O1',
# #  'E8': 'FP2',
# #  'E124': 'F4',
# #  'E103': 'C4',
# #  'E85': 'P4',
# #  'E83': 'O2',
# #  'E33': 'F7',
# #  'E45': 'T7',
# #  'E58': 'P7',
# #  'E122': 'F8',
# #  'E108': 'T8',
# #  'E96': 'P8',
# #  'E5': 'FZ',
# #  'E129': 'CZ',
# #  'E62': 'PZ'}

# # #!/usr/bin/env python3
# # # -*- coding: utf-8 -*-

# # import sys

# # import matplotlib.pyplot as plt
# # import mne
# # import numpy as np
# # import pandas as pd
# # from scipy.spatial import distance

# # # Functions
# # def cvt_to_1020_chs(raw):
# #     chs_df = pd.DataFrame(raw.get_montage().get_positions()["ch_pos"]).T
# #     STANDARD_MONTAGE = mne.channels.make_standard_montage("standard_1020")
# #     STANDARD_DF = pd.DataFrame(STANDARD_MONTAGE.get_positions()["ch_pos"]).T

# #     link_dict = {}
# #     for tgt_montage in CONFIG["TGT_MONTAGE"]:
# #         tgt_coord = STANDARD_DF.loc[tgt_montage.capitalize()]  # [REVISED]
# #         distances = distance.cdist(chs_df, np.array(tgt_coord).reshape(1, -1))  # [REVISED]
# #         min_dist = distances.min()
# #         i_min = np.argmin(distances)  # [REVISED]
# #         closest_ch = chs_df.index[i_min]  # [REVISED]
# #         link_dict[closest_ch] = tgt_montage

# #     raw.set_montage(mne.channels.make_standard_montage("standard_1020"))  # [REVISED]
# #     return raw  # [REVISED]

# # # Configurations
# # CONFIG = {'TGT_MONTAGE': ['Fz', 'Cz', 'Pz']}  # [REVISED]

# # # Load path
# # lpath = "./data/ds004635-download/sub-S9/eeg/sub-S9_task-resting_eeg.set"

# # # Main
# # raw = mne.io.read_raw_eeglab(lpath, preload=True)  # [REVISED]
# # raw = cvt_to_1020_chs(raw)
