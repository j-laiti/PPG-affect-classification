"""
AKTIVES Dataset TD Efficiency Training Time Evaluation
Table X in the paper
DOI: 10.1109/TAFFC.2025.3628467

Author: Justin Laiti
Note: Code organization and documentation assisted by Claude Sonnet 4.5
Last updated: Nov 15, 2025
"""
#%% imports
from curses import window
import numpy as np
import pandas as pd
import sys
import os
import time

# Add path for preprocessing functions
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from preprocessing.feature_extraction import *
from preprocessing.filters import *

# load data
wellby_data = pd.read_csv("../../data/Aktives/aktives_ppg_all_recordings.csv")
fs = 64 # Aktives sampling frequency (64 Hz)

#%% testing

all_td_features = []
recording_ids = []

# test first column
first_column = wellby_data.columns[0]
data = wellby_data[first_column]

# turn the rest of the column into a list to process
ppg_data = data.tolist()
ppg_data = ppg_data[1:] # skip first 2 seconds

ppg_data = data.dropna().tolist()
print(f"After removing NaN: {len(ppg_data)} data points")

# Apply preprocessing pipeline (same as hybrid approach)
bp_bvp = bandpass_filter(ppg_data, 0.2, 10, fs, order=2)
smoothed_signal = moving_average_filter(bp_bvp, window_size=5)

#%% define extraction function
def extract_td_features(data):
    """Extract time-domain features for the Aktives dataset"""

    all_td_features = []
    recording_ids = []
    
    # cycle through each column of data
    for col in data.columns:
        column_data = data[col]
        ppg_data = column_data.tolist()
        recording_ids.append(ppg_data[0])  # First entry is recording ID
        ppg_data = ppg_data[1:]

        # Remove NaN values
        ppg_data = [x for x in ppg_data if not pd.isna(x)]
        
        if len(ppg_data) == 0:
            continue

        # Apply preprocessing pipeline (same as hybrid approach)
        bp_bvp = bandpass_filter(ppg_data, 0.2, 10, fs, order=2)
        smoothed_signal = moving_average_filter(bp_bvp, window_size=5)
        
        segment_stds, std_ths = simple_dynamic_threshold(smoothed_signal, fs, 95, window_size=3)
        sim_clean_signal, clean_signal_indices = simple_noise_elimination(smoothed_signal, fs, std_ths)
        sim_final_clean_signal = moving_average_filter(sim_clean_signal, window_size=3)
        
        td_stats = get_ppg_features(ppg_seg=sim_final_clean_signal.tolist(), 
                                    fs=fs, 
                                    label=1, 
                                    calc_sq=True)
    
    # Store if successful
    if td_stats:
        all_td_features.append(td_stats)
                
    
    if all_td_features:
        features_df = pd.DataFrame(all_td_features)
        # features_df['recording_id'] = recording_ids
        return features_df

# run feature extraction
start_time = time.time()
features = extract_td_features(wellby_data)
end_time = time.time()

print(f"Time taken to extract TD features: {end_time - start_time} seconds")

# %%
