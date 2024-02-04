#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-04 16:19:51 (ywatanabe)"


import mne
import mngs
import numpy as np
import pandas as pd
from scipy.spatial import distance

CONFIG = mngs.io.load_configs(show=False)


def _rename_channels_for_consistency(raw, channel_mapping):
    raw.rename_channels(lambda x: x.upper().replace(".", ""))
    # Maps channels
    try:
        raw.rename_channels(channel_mapping)
    except:
        pass
    return raw


def _coords_to_1020(raw):
    def get__coords_to_1020_dict(chs_df, tgt_montages):
        STANDARD_MONTAGE = mne.channels.make_standard_montage("standard_1020")
        STANDARD_DF = pd.DataFrame(
            STANDARD_MONTAGE.get_positions()["ch_pos"]
        ).T

        link_dict = {}
        for tgt_montage in tgt_montages:
            tgt_coord = STANDARD_DF.T[tgt_montage.capitalize()]
            distances = ((chs_df - np.array(tgt_coord)[np.newaxis]) ** 2).sum(
                axis=1
            )
            min_dist = distances.min()
            i_min = np.where(distances == min_dist)[0][0]
            closest_ch = chs_df.iloc[i_min].name
            link_dict[closest_ch] = tgt_montage
        return link_dict

    try:
        coords_df = pd.DataFrame(raw.get_montage().get_positions()["ch_pos"]).T
    except Exception as e:
        print(e)
    cvt_dict = get__coords_to_1020_dict(coords_df, CONFIG["TGT_MONTAGE"])

    raw = raw.pick(list(cvt_dict.keys()), verbose=False)
    raw = raw.rename_channels(cvt_dict)
    return raw


def pick_channels(
    raw,
    tgt_montages=CONFIG["TGT_MONTAGE"],
):

    # Sets montages
    raw.set_montage(
        mngs.dsp.to_dig_montage(tgt_montages),
        on_missing="ignore",
        match_case=False,
    )

    try:
        # Picks predefined channels
        raw = raw.pick(CONFIG["TGT_MONTAGE"])
    except Exception as e:
        print(e)
        import ipdb

        ipdb.set_trace()
    return raw


def _check_has_tgt_montages(raw, channel_mapping, tgt_montages):
    has_tgt_motnages = False
    has_ch_names = raw.ch_names is not None
    if has_ch_names:
        raw = _rename_channels_for_consistency(raw, channel_mapping)
        tgt_montages_exp = [f"^{tm}" for tm in tgt_montages]
        has_tgt_montages = len(
            mngs.gen.search(tgt_montages_exp, raw.ch_names)[1]
        ) == len(tgt_montages)
    return has_tgt_montages


def _check_has_positions(raw):
    has_positions = False
    has_montage = raw.get_montage() is not None
    if has_montage:
        has_positions = raw.get_montage().get_positions()["ch_pos"] is not None
    return has_positions


def handle_channels(raw):
    try:
        # Check conditions
        has_positions = _check_has_positions(raw)
        if has_positions:
            raw = _coords_to_1020(raw)

        # When tgt montages are included
        has_tgt_montages = _check_has_tgt_montages(
            raw, CONFIG["CHANNEL_MAPPING"], CONFIG["TGT_MONTAGE"]
        )

        if has_tgt_montages:
            raw = pick_channels(raw, tgt_montages=CONFIG["TGT_MONTAGE"])
            loaded = raw
            # ch_names = raw.ch_names
            # print(f"\nLoaded:\n{raw.ch_names}")
        return raw
    except Exception as e:
        print(e)
        return None
