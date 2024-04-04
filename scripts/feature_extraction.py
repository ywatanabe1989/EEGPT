#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-03-30 17:09:39 (ywatanabe)"

import networkx as nx
import numpy as np
import torch
from fastdtw import fastdtw
from scipy.signal import hilbert
from scipy.spatial.distance import euclidean


def fft_amplitude_phase(eeg_data):
    """
    Calculates the amplitude and phase of an EEG signal using FFT.

    Parameters:
    - eeg_data (torch.Tensor): The EEG data as a PyTorch tensor.

    Returns:
    - tuple: A tuple containing the amplitude and phase of the EEG signal.
    """
    fft_vals = torch.fft.fft(eeg_data, dim=2)
    amplitude = torch.abs(fft_vals)
    phase = torch.angle(fft_vals)
    return amplitude, phase


# def hilbert_transform(eeg_data):
#     """
#     Applies the Hilbert transform to EEG data to obtain the amplitude envelope and instantaneous phase.

#     Parameters:
#     - eeg_data (numpy.ndarray): The EEG data as a NumPy array.

#     Returns:
#     - tuple: A tuple containing the amplitude envelope and instantaneous phase of the EEG signal.
#     """
#     analytic_signal = hilbert(eeg_data, axis=2)
#     amplitude_envelope = np.abs(analytic_signal)
#     instantaneous_phase = np.angle(analytic_signal)
#     return amplitude_envelope, instantaneous_phase


def power_spectrum(eeg_data):
    """
    Calculates the power spectrum density (PSD) of an EEG signal using FFT.

    Parameters:
    - eeg_data (torch.Tensor): The EEG data as a PyTorch tensor.

    Returns:
    - torch.Tensor: The power spectrum density of the EEG signal.
    """
    fft_vals = torch.fft.fft(eeg_data, dim=2)
    psd = torch.abs(fft_vals) ** 2
    return psd


def graph_analysis(adj_matrix):
    """
    Performs basic graph analysis on an adjacency matrix representing brain connectivity.

    Parameters:
    - adj_matrix (numpy.ndarray): The adjacency matrix representing brain connectivity.

    Returns:
    - dict: A dictionary containing average clustering coefficient, average path length, and centrality measures.
    """
    G = nx.from_numpy_matrix(adj_matrix)
    clustering_coeff = nx.average_clustering(G)
    path_length = nx.average_shortest_path_length(G)
    centrality = nx.degree_centrality(G)
    return {
        "clustering_coeff": clustering_coeff,
        "path_length": path_length,
        "centrality": centrality,
    }


from concurrent.futures import ProcessPoolExecutor


def dtw_distance(series1, series2):
    """
    Calculates the Dynamic Time Warping (DTW) distance between two time series.

    Parameters:
    - series1 (numpy.ndarray): The first time series.
    - series2 (numpy.ndarray): The second time series.

    Returns:
    - float: The DTW distance between the two time series.
    """
    distance, path = fastdtw(series1, series2, dist=euclidean)
    return distance


def calculate_dtw_distances(channels_data):
    """
    Calculates the DTW distances between all pairs of channels in parallel.

    Parameters:
    - channels_data (numpy.ndarray): The channels data as a NumPy array of shape (N_CHS, SEQ_LEN_PTS).

    Returns:
    - numpy.ndarray: A 2D array of DTW distances between all pairs of channels.
    """
    n_channels = channels_data.shape[0]
    dtw_distances = np.zeros((n_channels, n_channels))

    # Define a helper function for parallel execution
    def compute_dtw(pair):
        i, j = pair
        return i, j, dtw_distance(channels_data[i], channels_data[j])

    # Generate all unique pairs of channels to calculate DTW distance
    pairs = [
        (i, j) for i in range(n_channels) for j in range(i + 1, n_channels)
    ]

    # Use ProcessPoolExecutor to parallelize the DTW distance calculations
    with ProcessPoolExecutor() as executor:
        for i, j, distance in executor.map(compute_dtw, pairs):
            dtw_distances[i, j] = distance
            dtw_distances[j, i] = distance  # DTW distance is symmetric

    return dtw_distances


if __name__ == "__main__":
    BATCH_SIZE = 64
    N_CHS = 14
    SEQ_LEN_SEC = 1
    SAMP_RATE = 128
    SEQ_LEN_PTS = int(SEQ_LEN_SEC * SAMP_RATE)

    x = torch.randn(BATCH_SIZE, N_CHS, SEQ_LEN_PTS).cuda()

    amplitude, phase = fft_amplitude_phase(x)
    # amplitude_envelope, instantaneous_phase = hilbert_transform(x)
    psd = power_spectrum(x)
    # graph_analysis(x)
    # dtw_distance()

    # dtw_distances = calculate_dtw_distances(x[0])
