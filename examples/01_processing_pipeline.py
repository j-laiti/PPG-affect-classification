"""
PPG Signal Visualization Example - WESAD, AKTIVES, and Wellby Datasets

This script demonstrates the preprocessing pipeline on sample data from three datasets.
Generates Figure 3 from the manuscript.

Note: The Wellby dataset is not publicly available due to privacy restrictions.
      Examples using WESAD and AKTIVES datasets are fully reproducible.

Usage:
    - Run entire script: python visualize_ppg_preprocessing.py
    - Run interactively: Open in VSCode/Jupyter and execute cells with #%%
    
Author: Justin Laiti 
Note: Code organization and documentation assisted by Claude Sonnet 4.5
Last updated: Nov 15, 2025
"""

#%% Imports and Configuration
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from preprocessing.filters import (
    standardize, 
    bandpass_filter, 
    moving_average_filter,
    simple_dynamic_threshold,
    simple_noise_elimination
)
from preprocessing.peak_detection import threshold_peakdetection

# Font settings for publication-quality figures
TITLE_FONTSIZE = 20
LABEL_FONTSIZE = 18
TICK_FONTSIZE = 16

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times', 'Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'


#%% WESAD Dataset Example
# 
# Process and visualize PPG data from the WESAD dataset.
# Using Subject 2 (S2), time interval: 40.34 - 40.84 minutes
# Load and Prepare WESAD Data
def load_wesad_segment(file_path, start_min, end_min, fs=64):
    """
    Load a time segment from WESAD BVP data.
    
    Parameters
    ----------
    file_path : str
        Path to BVP.csv file
    start_min : float
        Start time in minutes
    end_min : float
        End time in minutes
    fs : int
        Sampling frequency in Hz
        
    Returns
    -------
    ppg_values : ndarray
        PPG signal values
    time_array : ndarray
        Time array in seconds (relative to segment start)
    """
    df = pd.read_csv(file_path)
    bvp_signal = df.iloc[:, 0].values
    
    # Create time column
    time_seconds = np.arange(len(bvp_signal)) / fs
    df["Time (s)"] = time_seconds
    
    # Select time interval
    start_time = start_min * 60
    end_time = end_min * 60
    df_filtered = df[(df["Time (s)"] >= start_time) & (df["Time (s)"] <= end_time)]
    
    # Extract and clean
    ppg_values = df_filtered.iloc[:, 0].values
    ppg_values = ppg_values[~np.isnan(ppg_values)]
    
    # Create relative time array
    time_array = np.arange(len(ppg_values)) / fs
    
    return ppg_values, time_array


# Load WESAD segment
WESAD_FILE = "../data/WESAD/S2/S2_E4_Data/BVP.csv" # TODO: Update path based on your setup
raw_ppg_wesad, time_wesad = load_wesad_segment(
    WESAD_FILE, 
    start_min=40.34, 
    end_min=40.84,
    fs=64
)

print(f"WESAD - Signal length: {len(raw_ppg_wesad)}")
print(f"WESAD - Time range: {time_wesad[0]:.2f} to {time_wesad[-1]:.2f} seconds")


#%% Process WESAD Signal
def process_ppg_signal(ppg_values, fs, lowcut=0.5, highcut=10.0, 
                       noise_percentile=85, order=2):
    """
    Apply full preprocessing pipeline to PPG signal.
    
    Parameters
    ----------
    ppg_values : ndarray
        Raw PPG signal
    fs : int
        Sampling frequency
    lowcut : float
        Bandpass filter lower cutoff
    highcut : float
        Bandpass filter upper cutoff
    noise_percentile : int
        Percentile threshold for noise elimination
    order : int
        Butterworth filter order
        
    Returns
    -------
    clean_signal : ndarray
        Processed PPG signal
    peaks : ndarray
        Indices of detected peaks
    """
    # Standardize
    signal = standardize(ppg_values)
    
    # Bandpass filter
    signal = bandpass_filter(signal, lowcut=lowcut, highcut=highcut, 
                            fs=fs, order=order)
    
    # Moving average
    signal = moving_average_filter(signal, window_size=5)
    
    # Noise elimination
    _, std_threshold = simple_dynamic_threshold(signal, fs, noise_percentile)
    clean_signal, _ = simple_noise_elimination(signal, fs, std_threshold)
    
    # Final smoothing
    clean_signal = moving_average_filter(clean_signal, window_size=5)
    
    # Peak detection
    peaks = threshold_peakdetection(clean_signal, fs)
    
    return clean_signal, peaks


# Process WESAD data
wesad_clean, wesad_peaks = process_ppg_signal(raw_ppg_wesad, fs=64)
wesad_clean_time = np.arange(len(wesad_clean)) / 64


#%% Plot WESAD Results
def plot_preprocessing_comparison(raw_signal, raw_time, clean_signal, 
                                 clean_time, peaks, title_prefix=""):
    """
    Create side-by-side comparison of raw and processed signals.
    
    Parameters
    ----------
    raw_signal : ndarray
        Raw PPG values
    raw_time : ndarray
        Time array for raw signal
    clean_signal : ndarray
        Processed PPG values
    clean_time : ndarray
        Time array for clean signal
    peaks : ndarray
        Peak indices in clean signal
    title_prefix : str
        Prefix for plot titles (e.g., "WESAD", "Wellby")
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    
    # Raw signal
    ax1.plot(raw_time, raw_signal, 'b-', linewidth=1)
    ax1.set_xlabel("Time (seconds)", fontsize=LABEL_FONTSIZE)
    ax1.set_ylabel("PPG Amplitude", fontsize=LABEL_FONTSIZE)
    ax1.set_title(f"{title_prefix} Raw PPG Signal", fontsize=TITLE_FONTSIZE)
    ax1.grid(True)
    ax1.tick_params(labelsize=TICK_FONTSIZE)
    
    # Processed signal
    ax2.plot(clean_time, clean_signal, 'g-', linewidth=1)
    ax2.scatter(clean_time[peaks], clean_signal[peaks], color='r', 
               label="Detected Peaks", zorder=5)
    ax2.set_xlabel("Time (seconds)", fontsize=LABEL_FONTSIZE)
    ax2.set_ylabel("PPG Amplitude", fontsize=LABEL_FONTSIZE)
    ax2.set_title(f"{title_prefix} Processed PPG Signal with Peak Detection", 
                 fontsize=TITLE_FONTSIZE)
    ax2.grid(True)
    ax2.tick_params(labelsize=TICK_FONTSIZE)
    
    plt.tight_layout()
    plt.show()


plot_preprocessing_comparison(
    raw_ppg_wesad, time_wesad,
    wesad_clean, wesad_clean_time, wesad_peaks,
    title_prefix="WESAD"
)

#%% AKTIVES Dataset Example
# Process PPG data from the AKTIVES dyslexia study using predefined analysis windows.

# Load AKTIVES Analysis Window
AKTIVES_WINDOWS = "../data/Aktives/analysis_windows/analysis_windows_dyslexia.csv" # TODO: Update path based on your setup
WINDOW_IDX = 10  # Example window

windows = pd.read_csv(AKTIVES_WINDOWS)
window = windows.iloc[WINDOW_IDX]

# Parse participant info
participant_cell = window["Participant"]
participant = participant_cell.split("_")[0]
game = participant_cell.split("_")[1]

interval_start = window["Interval_Start"]
interval_end = window["Interval_End"]
label = window["Label"]

print(f"\nAKTIVES - Participant: {participant}, Game: {game}")
print(f"Interval: {interval_start:.2f} - {interval_end:.2f}s, Label: {label}")


#%% Load and Extract AKTIVES PPG Segment
FS_AKTIVES = 64
aktives_file = f"../data/Aktives/PPG/Dyslexia/{participant}/{game}/BVP.csv"

df_aktives = pd.read_csv(aktives_file)
df_aktives['Time'] = df_aktives.index / FS_AKTIVES

# Handle comma decimal separator (European format)
df_aktives['values'] = (df_aktives['values']
                        .astype(str)
                        .str.replace(',', '.', regex=False)
                        .astype(float))

# Extract interval
interval_data = df_aktives[
    (df_aktives['Time'] >= interval_start) & 
    (df_aktives['Time'] <= interval_end)
]
raw_ppg_aktives = interval_data['values'].values
time_aktives = (interval_data['Time'].values - 
                interval_data['Time'].values[0])


#%% Process AKTIVES Signal
aktives_clean, aktives_peaks = process_ppg_signal(
    raw_ppg_aktives,
    fs=FS_AKTIVES,
    noise_percentile=90  # Higher threshold for this dataset
)
aktives_clean_time = np.arange(len(aktives_clean)) / FS_AKTIVES


#%% Plot AKTIVES Results
plot_preprocessing_comparison(
    raw_ppg_aktives, time_aktives,
    aktives_clean, aktives_clean_time, aktives_peaks,
    title_prefix="AKTIVES"
)

#%% Wellby Dataset Example
#
# Process and visualize PPG data from the Wellby study.
# Using a 30-second segment starting at 10 seconds.

# Load Wellby Data
WELLBY_FILE = '../data/Wellby/wellby_ppg_data.csv' # not publicly available
COLUMN_INDEX = 16  # Example column
FS_WELLBY = 50

ppg_data = pd.read_csv(WELLBY_FILE)
recording_id = ppg_data.columns[COLUMN_INDEX]

# Extract and clean
ppg_values = ppg_data.iloc[:, COLUMN_INDEX].values
ppg_values = ppg_values[~np.isnan(ppg_values)]

# Extract 30-second segment (10-40 seconds)
start_idx = 10 * FS_WELLBY
end_idx = start_idx + 30 * FS_WELLBY
raw_ppg_wellby = ppg_values[start_idx:end_idx]
time_wellby = np.arange(len(raw_ppg_wellby)) / FS_WELLBY

print(f"\nWellby - Recording: {recording_id}")
print(f"Wellby - Signal length: {len(raw_ppg_wellby)}")


#%% Process Wellby Signal
wellby_clean, wellby_peaks = process_ppg_signal(
    raw_ppg_wellby, 
    fs=FS_WELLBY,
    lowcut=0.7,
    highcut=5.0,
    noise_percentile=85
)
wellby_clean_time = np.arange(len(wellby_clean)) / FS_WELLBY


#%% Plot Wellby Results
plot_preprocessing_comparison(
    raw_ppg_wellby, time_wellby,
    wellby_clean, wellby_clean_time, wellby_peaks,
    title_prefix="Wellby"
)

