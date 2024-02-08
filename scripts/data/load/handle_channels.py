#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-05 21:53:54 (ywatanabe)"


import logging

import mne
import mngs
import numpy as np
import pandas as pd
from scipy.spatial import distance

CONFIG = mngs.io.load_configs(show=False)


def handle_channels(raw, verbose=False):
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
            raw = _pick_channels(raw, tgt_montages=CONFIG["TGT_MONTAGE"])
            loaded = raw

        # Ensure the channel names in 'raw' match the target montage
        assert set(raw.ch_names) == set(
            CONFIG["TGT_MONTAGE"]
        ), "The channel names in 'raw' do not match the target montage."

        if verbose:
            print(
                f"Congratulations! The files {[f for f in raw.filenames]} have been correctly channel-mapped."
            )

        return raw

    except Exception as e:
        logging.error(f"\nChannels were not properly handled:\n{e}")
        return None


def _map_channel_names(raw, channel_mapping):
    """
    Safely renames channels in a raw object based on a mapping,
    only applying changes for channels that exist in the raw object.
    If insufficient channels are found, excessive/insufficient channels are logged.

    Parameters:
    - raw (mne.io.Raw): The raw object containing EEG data.
    - channel_mapping (dict): A dictionary mapping old channel names to new names.
    """
    try:
        # To upper
        raw.rename_channels(lambda x: x.upper().replace(".", ""))

        valid_mapping = {}
        channel_mapping = CONFIG["CHANNEL_MAPPING"]
        for src, tgt in channel_mapping.items():
            if tgt in raw.ch_names:
                continue
            if src in raw.ch_names:
                valid_mapping[src] = tgt
        # Apply the valid renaming
        raw.rename_channels(valid_mapping)

        # Check excessive & insufficient channels for debugging
        excessive_channels = set(raw.ch_names) - set(CONFIG["TGT_MONTAGE"])
        insufficient_channels = set(CONFIG["TGT_MONTAGE"]) - set(raw.ch_names)
        if len(insufficient_channels):
            logging.error(f"\nExcessive Channels:\n{excessive_channels}")
            logging.error(f"\nInsufficient Channels:\n{insufficient_channels}")

        return raw

    except Exception as e:
        logging.error(f"\nChannels were not properly mapped:\n{e}")
        return None


def _coords_to_1020(raw):
    """
    Finds closest channels in the standard 10-20 system from coordinates.
    """

    def get_coords_to_1020_dict(chs_df, tgt_montages):
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
        cvt_dict = get_coords_to_1020_dict(coords_df, CONFIG["TGT_MONTAGE"])
        raw = raw.pick(list(cvt_dict.keys()), verbose=False)
        raw = raw.rename_channels(cvt_dict)
        return raw

    except Exception as e:
        logging.error(f"\Channels were not defined based on coordinates:\n{e}")
        return None


def _pick_channels(
    raw,
    tgt_montages=CONFIG["TGT_MONTAGE"],
):
    try:
        raw.set_montage(
            mngs.dsp.to_dig_montage(tgt_montages),
            on_missing="ignore",
            match_case=False,
        )
    except Exception as e:
        logging.error(f"\nSetting montage failed:\n{e}")

    try:
        raw = raw.pick(CONFIG["TGT_MONTAGE"])
    except Exception as e:
        logging.error(f"\nPicking channels failed:\n{e}")

    return raw


def _check_has_tgt_montages(raw, channel_mapping, tgt_montages):
    has_tgt_motnages = False
    has_ch_names = raw.ch_names is not None
    if has_ch_names:
        raw = _map_channel_names(raw, channel_mapping)
        # _print_insufficient_channels(raw.ch_names, tgt_montages)
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
        if has_positions:
            has_nans = (
                pd.DataFrame(raw.get_montage().get_positions()["ch_pos"])
                .isna()
                .any()
                .any()
            )
            if has_nans:
                has_positions = False
    return has_positions


def test_handle_channels():
    from scripts.utils.load_eeg_files import find_eeg_files
    from tqdm import tqdm

    LPATHS = find_eeg_files("OpenNeuro")

    lpaths_not_handled = []
    for i_lpath, lpath in tqdm(enumerate(np.random.permutation(LPATHS))):

        raw = mngs.io.load(lpath, verbose=False)
        if raw is None:
            lpaths_not_handled.append(lpath)
            logging.error(f"Not loaded: {lpath}")
            continue

        handled = handle_channels(raw)
        if handled is None:
            lpaths_not_handled.append(lpath)
            continue

        i_lpath += 1
        success_perc = 100 * ((i_lpath - len(lpaths_not_handled)) / i_lpath)
        if i_lpath % 10 == 0:
            print(f"Handled: {success_perc:.1f}%")


if __name__ == "__main__":
    test_handle_channels()  # more than 90% of files seem properly handled.
