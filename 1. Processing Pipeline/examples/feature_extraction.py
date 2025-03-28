import pandas as pd
import numpy as np
import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from processing_steps.processing_funcs import simple_statistic_adaptive_threshold, simple_noise_elimination, simple_bandpassfilter, bandpass_filter, movingaverage, eliminate_noise_in_time, normalize, statistic_threshold
from feature_calc.metric_calc_funcs import get_ppg_features

def calculate_metrics(raw_ppg):
    start_index = 500
    delayed_ppg = raw_ppg[start_index:]

    fs = 50  # Sampling frequency in Hz
    cycle = 5

    # Apply bandpass filter
    bp_ppg = bandpass_filter(delayed_ppg, 0.7, 5, fs)
    # Apply moving average filter
    fwd = movingaverage(bp_ppg, window_size=3)
    bwd = movingaverage(bp_ppg[::-1], window_size=3)
    smoothed_ppg = np.mean(np.vstack((fwd, bwd[::-1])), axis=0)
    #normalize data
    norm_PPG = normalize(smoothed_ppg)
    # Apply denoising filter
    temp_ths = [1.0, 2.0, 1.8, 1.5]
    signal_5_percent = int(len(norm_PPG) * 0.1)  # Use first 0.1% of the signal
    clean_signal = norm_PPG[:signal_5_percent]  # Assumed clean
    print(f"Clean signal (first 0.1%): {clean_signal}")
    print(f"Clean signal length: {len(clean_signal)}")
    thresholds = statistic_threshold(clean_signal, fs, temp_ths)
    print(f"Dynamic thresholds calculated: {thresholds}")
    len_before, len_after, clean_indices = eliminate_noise_in_time(norm_PPG, fs, thresholds, cycle=cycle)
    # Check if clean_indices is empty
    if len(clean_indices) == 0:
        print("No clean data detected. Adjust thresholds or inspect the input signal.")
        return {}
    cleaned_ppg = smoothed_ppg[clean_indices]
    # Apply a second smoothing filter
    final_smoothed_ppg = movingaverage(cleaned_ppg, window_size=3)

    # Get PPG features and signal quality
    window_length = len(final_smoothed_ppg) // fs
    ppg_features = get_ppg_features(final_smoothed_ppg, window_length=window_length, fs=fs, ensemble=True)

    if not ppg_features:
        print(f"No valid HRV metrics calculated for participant.")
        return {}

    return ppg_features

def simplified_calculate_metrics(raw_ppg):
    start_index = 500
    delayed_ppg = raw_ppg[start_index:]

    fs = 50  # Sampling frequency in Hz

    b = [ 0.05223535, 0.0, -0.1044707, 0.0, 0.05223535]
    a = [ 1.0, -3.16573522, 3.83025017, -2.12887887, 0.46652974]

    # Apply bandpass filter
    bp_ppg = simple_bandpassfilter(b, a, delayed_ppg)

    # Apply moving average filter
    smoothed_ppg = movingaverage(bp_ppg, window_size=3)

    std_ths = simple_statistic_adaptive_threshold(smoothed_ppg, 0.5)
    noise_eliminated_clean_indices = simple_noise_elimination(smoothed_ppg, fs, std_ths, window_size=3)
    
    if len(noise_eliminated_clean_indices) == 0:
        print("No clean data detected. Adjust thresholds or inspect the input signal.")
        return {}
    cleaned_ppg = smoothed_ppg[noise_eliminated_clean_indices]
    # Apply a second smoothing filter
    final_smoothed_ppg = movingaverage(cleaned_ppg, window_size=3)

    # Get PPG features and signal quality
    window_length = len(final_smoothed_ppg) // fs
    ppg_features = get_ppg_features(final_smoothed_ppg, window_length=window_length, fs=fs, ensemble=True)

    if not ppg_features:
        print(f"No valid HRV metrics calculated for participant.")
        return {}

    return ppg_features



#%% single rec calc

# # Run the calculation function
# recording_csv = 'data/raw_ppg_data.csv'
# data = pd.read_csv(recording_csv)
# ppg_signal = data["85A3F0D7-04AC-485F-9B00-1B90B99672D8"].dropna().values
# calculate_metrics(ppg_signal)


#%% mass recording feature extraction

recording_csv = '../data/raw_data/selected_ppg_data.csv'
data = pd.read_csv(recording_csv)

features_list = []
columns = ['Session_ID']

for session_id in data.columns:
    print(f"Processing session: {session_id}")
    ppg_values = data[session_id].dropna().values
    
    # Skip sessions with no valid data
    if len(ppg_values) == 0:
        print(f"No data for session {session_id}, skipping.")
        continue

    features = simplified_calculate_metrics(ppg_values)
    
    # Skip sessions where no features are extracted
    if not features:
        print(f"No features extracted for session {session_id}, skipping.")
        continue

    # Add session ID and features to the list
    if len(columns) == 1:  # Initialize columns on the first valid session
        columns += list(features.keys())

    features_list.append([session_id] + list(features.values()))

# Convert the features list to a DataFrame
features_df = pd.DataFrame(features_list, columns=columns)

# Save the features to a CSV file
output_csv = 'extracted_features.csv'
features_df.to_csv(output_csv, index=False)

print(f"Features extracted and saved to {output_csv}")


# %%
