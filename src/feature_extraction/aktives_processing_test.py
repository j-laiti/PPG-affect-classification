# just an initial test to select data from the "analysis_window" files
# and then select the corresponding PPG data and visualize it


#%% imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from preprocessing.filters import *
from preprocessing.peak_detection import *
from preprocessing.feature_extraction import *

#%% import analysis window define relevant data

windows = pd.read_csv("../..//data/Aktives/analysis_windows/analysis_windows_dyslexia.csv")
# select first row to start with
first_row = windows.iloc[0]

participant_cell = first_row["Participant"]
participant = participant_cell.split("_")[0]
game = participant_cell.split("_")[1]
print(f"Game: {game}, Participant: {participant}")

interval_start = first_row["Interval_Start"]
interval_end = first_row["Interval_End"]
label = first_row["Label"]

fs = 64  # Sampling frequency in Hz
# %% now lets load the PPG data for this participant and game

relevant_ppg_data = pd.read_csv(f"../../data/Aktives/PPG/Dyslexia/{participant}/{game}/BVP.csv")

# add a time column (in seconds from zero)
relevant_ppg_data["Time"] = relevant_ppg_data.index / 64
time_segment = relevant_ppg_data["Time"].values
raw_signal_relative_time = time_segment - time_segment[0]
# %% now plot!

# Ensure all values are strings, then replace comma with dot, then convert to float
relevant_ppg_data['values'] = relevant_ppg_data['values'].astype(str).str.replace(',', '.', regex=False).astype(float)

# Now select the interval and plot
ppg_interval = relevant_ppg_data[(relevant_ppg_data['Time'] >= interval_start) & (relevant_ppg_data['Time'] <= interval_end)]
raw_ppg_values = ppg_interval['values'].values

plt.figure(figsize=(12, 6))
plt.plot(ppg_interval['Time'], ppg_interval['values'], label='BVP Signal')
plt.title(f'PPG Data for {participant} - {game}\nInterval: {interval_start} to {interval_end} seconds\nLabel: {label}')
plt.xlabel('Time (seconds)')
plt.ylabel('BVP Signal')
plt.legend()
plt.grid()
plt.show()

# %% now lets try processing this segment of data

# Remove NaN and standardize
# raw_ppg_values = ppg_interval[~np.isnan(ppg_interval)]
ppg_standardized = standardize(raw_ppg_values) 
time_raw = raw_signal_relative_time[:len(raw_ppg_values)]

print(f"Original signal length: {len(raw_ppg_values)}")
# print(f"After NaN removal: {len(ppg_interval)}")
print(f"Standardized signal stats - Mean: {np.mean(ppg_standardized):.3f}, Std: {np.std(ppg_standardized):.3f}")
# %%
# Apply filters step by step

# Step 1: Bandpass filter
bandpass_signal = bandpass_filter(ppg_standardized, lowcut=0.5, highcut=10.0, fs=fs, order=2)

# Step 2: Moving average smoothing
smoothed_signal = moving_average_filter(bandpass_signal, window_size=5)

# Step 3: Simple dynamic thresholding and noise elimination
segment_stds, std_ths = simple_dynamic_threshold(smoothed_signal, fs, 95)
clean_signal, clean_indices = simple_noise_elimination(smoothed_signal, fs, std_ths)

# Step 4: Final smoothing
final_clean_signal = moving_average_filter(clean_signal, window_size=3)

# Map clean signal to time (basically just create an array as long as the clean signal?)
clean_time_mapped = np.arange(len(final_clean_signal)) / fs

# Convert processed signal into DataFrame for easier plotting
processed_df = pd.DataFrame({"Time (s)": clean_time_mapped, "PPG": final_clean_signal})

# run peak detection
peaks_threshold = threshold_peakdetection(final_clean_signal, fs)
peak_times = processed_df["Time (s)"].iloc[peaks_threshold].values

# %%
#%% Plot Raw vs Final Clean Signal

# Font settings for figures
title_fontsize = 18
label_fontsize = 16
tick_fontsize = 14

plt.figure(figsize=(15, 8))

# Top subplot: Raw signal
plt.subplot(2, 1, 1)
plt.plot(time_raw, raw_ppg_values, 'b-', linewidth=1, label="Raw PPG Signal")
plt.xlabel("Time", fontsize=label_fontsize)
plt.ylabel("PPG Amplitude", fontsize=label_fontsize)
plt.title("Raw PPG Signal", fontsize=title_fontsize)
plt.grid(True)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.legend()

# Bottom subplot: Final clean signal
plt.subplot(2, 1, 2)
plt.plot(processed_df["Time (s)"], processed_df["PPG"], 'g-', linewidth=1, label="Final Clean Signal")
plt.scatter(peak_times, final_clean_signal[peaks_threshold], color='r', label="Detected Peaks")
plt.xlabel("Time", fontsize=label_fontsize)
plt.ylabel("PPG Amplitude", fontsize=label_fontsize)
plt.title("Final Clean Signal (After Full Pipeline)", fontsize=title_fontsize)
plt.legend()
plt.grid(True)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.tight_layout()
plt.show()

# %% using those peaks, extract the time domain HRV data
#   # this spe
ppg_features = get_ppg_features(final_clean_signal, fs, label, ppg_standardized, calc_sq=True)

# %%
