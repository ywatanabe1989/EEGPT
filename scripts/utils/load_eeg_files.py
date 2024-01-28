#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-01-29 08:04:02 (ywatanabe)"

import random

import mne
import mngs
import numpy as np
import pandas as pd
from scipy.spatial import distance

CONFIG = mngs.io.load_configs()

# Functions
def rename_channels_for_consistency(raw, channel_mapping):
    raw.rename_channels(lambda x: x.upper().replace(".", ""))
    # Maps channels
    try:
        raw.rename_channels(channel_mapping)
    except:
        pass
    return raw


def coords_to_1020(raw):
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
    except Exception as e:
        print(e)
        import ipdb

        ipdb.set_trace()
    cvt_dict = get_coords_to_1020_dict(coords_df, CONFIG["TGT_MONTAGE"])
    raw = raw.pick(list(cvt_dict.keys()))
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


def check_has_tgt_montages(raw, channel_mapping, tgt_montages):
    has_tgt_motnages = False
    has_ch_names = raw.ch_names is not None
    if has_ch_names:
        raw = rename_channels_for_consistency(raw, channel_mapping)
        tgt_montages_exp = [f"^{tm}" for tm in tgt_montages]
        has_tgt_montages = len(
            mngs.gen.search(tgt_montages_exp, raw.ch_names)[1]
        ) == len(tgt_montages)
    return has_tgt_montages


def check_has_positions(raw):
    has_positions = False
    has_montage = raw.get_montage() is not None
    if has_montage:
        has_positions = raw.get_montage().get_positions()["ch_pos"] is not None
    return has_positions


def load_an_eeg_file(lpath):
    loaded = None
    ch_names = None
    unloaded = None
    has_positions = None
    has_tgt_montages = None

    try:
        raw = mngs.io.load(lpath, verbose=False)

        # Check conditions
        has_positions = check_has_positions(raw)
        if has_positions:
            raw = coords_to_1020(raw)

        # When tgt montages are included
        has_tgt_montages = check_has_tgt_montages(
            raw, CONFIG["CHANNEL_MAPPING"], CONFIG["TGT_MONTAGE"]
        )

        if has_tgt_montages:
            raw = pick_channels(raw, tgt_montages=CONFIG["TGT_MONTAGE"])
            loaded = raw
            ch_names = raw.ch_names
            print(f"\nLoaded:\n{raw.ch_names}")

    except Exception as e:
        print(e)
        unloaded = (lpath, e)

    return loaded  # , ch_names, unloaded, has_positions, has_tgt_montages


def load_eeg_files(lpaths, n_loads=None):
    print(f"\nLoading from {n_loads} of {len(lpaths)} files")

    if n_loads is not None:
        lpaths = np.random.permutation(lpaths)[:n_loads]

    loaded = []
    # ch_names = []
    # unloaded = []
    # has_positions = []
    # has_tgt_montages = []

    for lpath in lpaths:
        # (
        #     _loaded,
        #     _ch_names,
        #     _unloaded,
        #     _has_positions,
        #     _has_tgt_montages,
        # ) = load_an_eeg_file(lpath)

        _loaded = load_an_eeg_file(lpath)

        if _loaded is not None:
            loaded.append(_loaded)
        # ch_names.append(_ch_names)
        # unloaded.append(_unloaded)
        # has_positions.append(_has_positions)
        # has_tgt_montages.append(_has_tgt_montages)

    # unloaded = pd.DataFrame(unloaded)
    print(f"\nLoaded: {len(loaded) / n_loads * 100}% of {n_loads} data")
    return loaded  # , ch_names, unloaded


def preprocess_raw(raw):
    # Re-reference
    try:
        raw = mne.set_bipolar_reference(
            raw,
            anode=[
                bi.split("-")[0]
                for bi in CONFIG["TGT_TRANVERSE_BIPOLAR_MONTAGE"]
            ],
            cathode=[
                bi.split("-")[1]
                for bi in CONFIG["TGT_TRANVERSE_BIPOLAR_MONTAGE"]
            ],
            ch_name=CONFIG["TGT_TRANVERSE_BIPOLAR_MONTAGE"],
            drop_refs=True,
        )
    except Exception as e:
        print(e)
        return None

    # Notch filtering at 50 and 60 Hz
    raw = raw.notch_filter(freqs=[50, 60], n_jobs=1)

    # Interpolate bad channels if any
    raw = raw.interpolate_bads(verbose=False)

    # Downsampling to CONFIG["TGT_FS"]
    raw.resample(CONFIG["TGT_FS"], n_jobs=1)
    return raw


# def segment_raw(raw, CONFIG):
def segment_raw(raw):
    # Convert to DataFrame and transpose
    df = raw.to_data_frame().set_index("time").T

    # Segmentation
    segs = mngs.dsp.crop(
        np.array(df),
        int(CONFIG["WINDOW_SIZE_SEC"] * CONFIG["TGT_FS"]),  # 256
        overlap_factor=CONFIG["OVERLAP_FACTOR"],
    )

    # Limit the number of segments to reduce sampling bias
    segs = segs[
        np.random.permutation(np.arange(len(segs)))[
            : CONFIG["MAX_N_WINDOWS_PER_SUB"]
        ]
    ]
    return segs


def load_processed_eeg_file(filepath):
    with mngs.gen.suppress_output():
        try:
            raw = load_an_eeg_file(filepath)
        except Exception as e:
            print(f"\nError:\n{filepath} was not loaded.\n{e}")

        if raw is not None:
            try:
                raw = preprocess_raw(raw)
            except Exception as e:
                print(f"\nError:\n{filepath} was not preprocessed.\n{e}")

            try:
                segs = segment_raw(raw)
            except Exception as e:
                print(f"\nError:\n{filepath} was not segmented.\n{e}")

            return segs
        else:
            return None


if __name__ == "__main__":
    # LPATHS = mngs.io.load(CONFIG["EEG_FILES_LIST_OPENNEURO"])
    LPATHS = mngs.io.load(CONFIG["EEG_FILES_LIST_INTERNAL"])
    N_LOADS = 100
    loaded = load_eeg_files(LPATHS, n_loads=N_LOADS)
    processed = load_processed_eeg_file(LPATHS[0])
