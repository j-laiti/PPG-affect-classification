# The purpose of this file is to create graphs of the PPG data for WESAD and Wellby through different stages of preprocessing
# Figure 3 in the manuscript was created using this code.
# Justin Laiti June 22 2025

#%% imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from preprocessing.filters import *
from preprocessing.peak_detection import *

# Font settings for figures
title_fontsize = 20
label_fontsize = 18
tick_fontsize = 16

#%% Visualize the preprocesing of a segment of data from the WESAD dataset

# WESAD BVP file path. For this example, data is chosen from S2
file_path = "../data/WESAD/S2/S2_E4_Data/BVP.csv"

# Load dataset
df = pd.read_csv(file_path)
bvp_signal = df.iloc[:, 0].values

# Sampling frequency (Hz)
fs = 64  
time_seconds = np.arange(len(bvp_signal)) / fs  # Generate time column in seconds
df["Time (s)"] = time_seconds

# Select time interval of focus (this example highlights PPG data between 40.12 - 41.12 minutes)
start_time = 40.34 * 60  # Convert to seconds
end_time = 40.84 * 60
df_filtered = df[(df["Time (s)"] >= start_time) & (df["Time (s)"] <= end_time)]

# Extract PPG values
ppg_values = df_filtered.iloc[:, 0].values
filtered_time_segment = df_filtered["Time (s)"].values
raw_signal_relative_time = filtered_time_segment - filtered_time_segment[0]

# Remove NaN and standardize
raw_ppg_values = ppg_values[~np.isnan(ppg_values)]
ppg_standardized = standardize(raw_ppg_values) 
time_raw = raw_signal_relative_time[:len(raw_ppg_values)]  # Match time to cleaned signal

print(f"Original signal length: {len(ppg_values)}")
print(f"After NaN removal: {len(raw_ppg_values)}")
print(f"Standardized signal stats - Mean: {np.mean(ppg_standardized):.3f}, Std: {np.std(ppg_standardized):.3f}")

#%% Apply filters step by step

# Step 1: Bandpass filter
bandpass_signal = bandpass_filter(ppg_standardized, lowcut=0.5, highcut=10.0, fs=fs, order=2)

# Step 2: Moving average smoothing
smoothed_signal = moving_average_filter(bandpass_signal, window_size=5)

# Step 3: Simple dynamic thresholding and noise elimination
segment_stds, std_ths = simple_dynamic_threshold(smoothed_signal, fs, 85)
clean_signal, clean_indices = simple_noise_elimination(smoothed_signal, fs, std_ths)

# Step 4: Final smoothing
final_clean_signal = moving_average_filter(clean_signal, window_size=5)

# Map clean signal to time (basically just create an array as long as the clean signal?)
clean_time_mapped = np.arange(len(final_clean_signal)) / fs

# Convert processed signal into DataFrame for easier plotting
processed_df = pd.DataFrame({"Time (s)": clean_time_mapped, "PPG": final_clean_signal})

# run peak detection
peaks_threshold = threshold_peakdetection(final_clean_signal, fs)
peak_times = processed_df["Time (s)"].iloc[peaks_threshold].values

#%% Plot Raw vs Final Clean Signal

plt.figure(figsize=(8, 8))

# Top subplot: Raw signal
plt.subplot(2, 1, 1)
plt.plot(time_raw, raw_ppg_values, 'b-', linewidth=1, label="Raw PPG Signal")
plt.xlabel("Time (seconds)", fontsize=label_fontsize, fontfamily='serif')
plt.ylabel("PPG Amplitude", fontsize=label_fontsize, fontfamily='serif')
plt.title("Raw PPG Signal", fontsize=title_fontsize, fontfamily='serif')
plt.grid(True)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
# plt.legend()

# Bottom subplot: Final clean signal
plt.subplot(2, 1, 2)
plt.plot(processed_df["Time (s)"], processed_df["PPG"], 'g-', linewidth=1, label="Final Clean Signal")
plt.scatter(peak_times, final_clean_signal[peaks_threshold], color='r', label="Detected Peaks")
plt.xlabel("Time (seconds)", fontsize=label_fontsize, fontfamily='serif')
plt.ylabel("PPG Amplitude", fontsize=label_fontsize, fontfamily='serif')
plt.title("Processed PPG Signal with Peak Detection", fontsize=title_fontsize, fontfamily='serif')
plt.grid(True)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.tight_layout()
plt.show()


#%% comparison of bandpass filter ranges










#%% Visualize the preprocesing of a segment of data from the Wellby dataset

file_path = '../data/Wellby/selected_ppg_data.csv'
ppg_data = pd.read_csv(file_path)

# Select column for processing
column_index = 16  # Update this to choose a new column
ppg_values = ppg_data.iloc[:, column_index].values
recording_id = ppg_data.columns[column_index]  # Get column name as ID

# Remove NaN values
raw_ppg_values = ppg_values[~np.isnan(ppg_values)]  

# Define sampling frequency
fs = 50  

# focus on a 30 sec segment of the data
start_index = 10 *fs # Start at 10 seconds
end_index = start_index + 30 * fs  # 30 seconds segment
ppg_values = raw_ppg_values[start_index:end_index]

# Standardize signal
ppg_values = standardize(ppg_values)  

# Create **original** time axis based on the entire segment
time_seconds = np.arange(len(raw_ppg_values)) / fs  

# Ensure **time array matches the processed signal length**
filtered_time_seconds = time_seconds[start_index:start_index + len(ppg_values)]

#%% Run Filtering & Peak Detection

# Apply filters
wellby_filtered_signal = bandpass_filter(ppg_values, lowcut=0.7, highcut=5.0, fs=fs, order=2)
wellby_smoothed_signal = moving_average_filter(wellby_filtered_signal, window_size=5)

# Noise elimination
segment_stds, std_ths = simple_dynamic_threshold(wellby_smoothed_signal, fs, 85)
wellby_clean_signal, wellby_clean_indices = simple_noise_elimination(wellby_smoothed_signal, fs, std_ths)

# final smoothing
wellby_final_clean_signal = moving_average_filter(wellby_clean_signal, window_size=3)

# Ensure **time axis matches cleaned signal**
processed_time_seconds = np.arange(len(wellby_final_clean_signal)) / fs

# Detect peaks & adjust their locations
sim_peaks_threshold = threshold_peakdetection(wellby_final_clean_signal, fs)
peak_times = processed_time_seconds[sim_peaks_threshold]  # Use correct time values


# %% Plot Processed PPG with Peak Detection

# Set Times font at the beginning
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times', 'Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'

plt.figure(figsize=(8, 8))

# Top subplot: Raw signal
plt.subplot(2, 1, 1)
plt.plot(filtered_time_seconds, ppg_values, 'b-', linewidth=1, label="Raw PPG Signal")
plt.xlabel("Time (seconds)", fontsize=label_fontsize, fontfamily='serif')
plt.ylabel("PPG Amplitude", fontsize=label_fontsize, fontfamily='serif')
plt.title("Raw PPG Signal", fontsize=title_fontsize, fontfamily='serif')
plt.grid(True)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
# plt.legend()

# Bottom subplot: Final clean signal
plt.subplot(2, 1, 2)
plt.plot(processed_time_seconds, wellby_final_clean_signal, 'g-', linewidth=1, label="Final Clean Signal")
plt.scatter(peak_times, wellby_final_clean_signal[sim_peaks_threshold], color='r', label="Detected Peaks")
plt.xlabel("Time (seconds)", fontsize=label_fontsize, fontfamily='serif')
plt.ylabel("PPG Amplitude", fontsize=label_fontsize, fontfamily='serif')
plt.title("Processed PPG Signal with Peak Detection", fontsize=title_fontsize, fontfamily='serif')
plt.grid(True)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.tight_layout()
plt.show()







#%% aktive data

# import analysis window define relevant data

windows = pd.read_csv("../data/Aktives/analysis_windows/analysis_windows_dyslexia.csv")
# select first row to start with
first_row = windows.iloc[10]

participant_cell = first_row["Participant"]
participant = participant_cell.split("_")[0]
game = participant_cell.split("_")[1]
print(f"Game: {game}, Participant: {participant}")

interval_start = first_row["Interval_Start"]
interval_end = first_row["Interval_End"]
label = first_row["Label"]

fs = 64  # Sampling frequency in Hz
# %% now lets load the PPG data for this participant and game

relevant_ppg_data = pd.read_csv(f"../data/Aktives/PPG/Dyslexia/{participant}/{game}/BVP.csv")

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
segment_stds, std_ths = simple_dynamic_threshold(smoothed_signal, fs, 90)
clean_signal, clean_indices = simple_noise_elimination(smoothed_signal, fs, std_ths)

# Step 4: Final smoothing
final_clean_signal = moving_average_filter(clean_signal, window_size=5)

# Map clean signal to time (basically just create an array as long as the clean signal?)
clean_time_mapped = np.arange(len(final_clean_signal)) / fs

# Convert processed signal into DataFrame for easier plotting
processed_df = pd.DataFrame({"Time (s)": clean_time_mapped, "PPG": final_clean_signal})

# run peak detection
peaks_threshold = threshold_peakdetection(final_clean_signal, fs)
peak_times = processed_df["Time (s)"].iloc[peaks_threshold].values

#%% Plot Raw vs Final Clean Signal

# Font settings for figures
title_fontsize = 20
label_fontsize = 18
tick_fontsize = 16

plt.figure(figsize=(8, 8))

# Top subplot: Raw signal
plt.subplot(2, 1, 1)
plt.plot(time_raw, raw_ppg_values, 'b-', linewidth=1, label="Raw PPG Signal")
plt.xlabel("Time (seconds)", fontsize=label_fontsize, fontfamily='serif')
plt.ylabel("PPG Amplitude", fontsize=label_fontsize, fontfamily='serif')
plt.title("Raw PPG Signal", fontsize=title_fontsize, fontfamily='serif')
plt.grid(True)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
# plt.legend()

# Bottom subplot: Final clean signal
plt.subplot(2, 1, 2)
plt.plot(processed_df["Time (s)"], processed_df["PPG"], 'g-', linewidth=1, label="Final Clean Signal")
plt.scatter(peak_times, final_clean_signal[peaks_threshold], color='r', label="Detected Peaks")
plt.xlabel("Time (seconds)", fontsize=label_fontsize, fontfamily='serif')
plt.ylabel("PPG Amplitude", fontsize=label_fontsize, fontfamily='serif')
plt.title("Processed PPG Signal with Peak Detection", fontsize=title_fontsize, fontfamily='serif')
# plt.legend()
plt.grid(True)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.tight_layout()
plt.show()

# %% using those peaks, extract the time domain HRV data
#   # this spe
ppg_features = get_ppg_features(final_clean_signal, fs, label, ppg_standardized, calc_sq=True)

# %%
