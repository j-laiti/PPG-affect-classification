"""
WESAD Dataset Processing Pipeline

Processes the WESAD dataset for stress classification using PPG signals.
Implements bandpass filtering, noise elimination, peak detection, and time-domain
HRV feature extraction. Adapted from Heo et al. (2021).

Reference: https://doi.org/10.1109/ACCESS.2021.3060441

Usage:
    python wesad_td_ppg_extraction.py
    
Author: Justin Laiti (adapted from Heo et al., 2021)
Note: Code organization and documentation assisted by Claude Sonnet 4.5
Last updated: Nov 15, 2025
"""

#%% Imports
import os
import pickle
import time
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

from preprocessing.feature_extraction import *
from preprocessing.filters import *

#%% Configuration
WINDOW_IN_SECONDS = 120
NOISE = ['bp_time']
main_path = '../../data/WESAD/'

# Sampling frequencies for E4 (wrist) sensors
fs_dict = {
    'ACC': 32, 
    'BVP': 64, 
    'EDA': 4, 
    'TEMP': 4, 
    'label': 700, 
    'Resp': 700
}

# Label mappings
label_dict = {'baseline': 0, 'stress': 1, 'amusement': 0}
int_to_label = {0: 'baseline', 1: 'stress', 2: 'amusement'}

# Processing parameters
sec = 12
N = fs_dict['BVP'] * sec
overlap = int(np.round(N * 0.02))
overlap = overlap if overlap % 2 == 0 else overlap + 1

#%% Helper Classes and Functions

class SubjectData:
    """Load and access WESAD subject data."""
    
    def __init__(self, main_path, subject_number):
        self.name = f'S{subject_number}'
        self.subject_keys = ['signal', 'label', 'subject']
        self.signal_keys = ['chest', 'wrist']
        self.chest_keys = ['ACC', 'ECG', 'EMG', 'EDA', 'Temp', 'Resp']
        self.wrist_keys = ['ACC', 'BVP', 'EDA', 'TEMP']
        
        with open(os.path.join(main_path, self.name, f'{self.name}.pkl'), 'rb') as file:
            self.data = pickle.load(file, encoding='latin1')
        self.labels = self.data['label']

    def get_wrist_data(self):
        """Extract wrist sensor data plus respiration."""
        data = self.data['signal']['wrist'].copy()
        data.update({'Resp': self.data['signal']['chest']['Resp']})
        return data


def extract_ppg_data(e4_data_dict, labels, norm_type=None):
    """Extract and normalize PPG data with labels."""
    df = pd.DataFrame(e4_data_dict['BVP'], columns=['BVP'])
    label_df = pd.DataFrame(labels, columns=['label'])
    
    # Set time indices
    df.index = [(1 / fs_dict['BVP']) * i for i in range(len(df))]
    label_df.index = [(1 / fs_dict['label']) * i for i in range(len(label_df))]
    
    # Convert to datetime
    df.index = pd.to_datetime(df.index, unit='s')
    label_df.index = pd.to_datetime(label_df.index, unit='s')
    
    df = df.join(label_df, how='outer')
    df['label'] = df['label'].bfill()
    df.reset_index(drop=True, inplace=True)
    
    if norm_type == 'std':
        df['BVP'] = (df['BVP'] - df['BVP'].mean()) / df['BVP'].std()
    elif norm_type == 'minmax':
        df = (df - df.min()) / (df.max() - df.min())
    
    df = df.dropna(axis=0)
    return df


def seperate_data_by_label(df):
    """Split data by baseline/stress/amusement labels."""
    grouped = df.groupby('label')
    baseline = grouped.get_group(1)
    stress = grouped.get_group(2)
    amusement = grouped.get_group(3)
    return grouped, baseline, stress, amusement


def get_samples(data, label, ma_usage):
    """Extract sliding window samples and compute HRV features."""
    samples = []
    
    window_len = fs_dict['BVP'] * WINDOW_IN_SECONDS
    sliding_window_len = int(fs_dict['BVP'] * WINDOW_IN_SECONDS * 0.25)
    
    i = 0
    while sliding_window_len * i <= len(data) - window_len:
        w = data[sliding_window_len * i : (sliding_window_len * i) + window_len]
        wstats = get_ppg_features(
            ppg_seg=w['BVP'].tolist(), 
            fs=fs_dict['BVP'], 
            label=label, 
            calc_sq=False
        )
        
        if wstats:
            samples.append(pd.DataFrame(wstats, index=[i]))
        i += 1
    
    return pd.concat(samples) if samples else pd.DataFrame()


def combine_files(subjects):
    """Combine processed subject files into single dataset."""
    df_list = []
    for s in subjects:
        df = pd.read_csv(f'{savePath}{subject_feature_path}/S{s}.csv', index_col=0)
        df['subject'] = s
        df_list.append(df)
    
    df = pd.concat(df_list)
    df['label'] = df[['0', '1']].idxmax(axis=1).astype(int)
    df.drop(['0', '1'], axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(savePath + merged_path)
    
    counts = df['label'].value_counts()
    print('Number of samples per class:')
    for label, number in zip(counts.index, counts.values):
        print(f'{int_to_label[label]}: {number}')


def make_patient_data(subject_id, ma_usage):
    """Process single subject's data through full pipeline."""
    subject = SubjectData(main_path=main_path, subject_number=subject_id)
    e4_data_dict = subject.get_wrist_data()
    
    df = extract_ppg_data(e4_data_dict, subject.labels, norm_type='std')
    df_BVP = df.BVP.tolist()
    
    # Apply filters
    bp_bvp = bandpass_filter(df_BVP, 0.5, 10, fs_dict['BVP'], order=2)
    
    if BP:
        df['BVP'] = bp_bvp
    
    if TIME:
        smoothed_signal = moving_average_filter(bp_bvp, window_size=5)
        df['BVP'] = smoothed_signal
        
        segment_stds, std_ths = simple_dynamic_threshold(
            smoothed_signal, fs_dict['BVP'], 85, window_size=3
        )
        
        print(f"Dynamic Threshold (95th Percentile): {std_ths}")
        print(f"Segment Standard Deviations: {segment_stds}")
        
        sim_clean_signal, clean_signal_indices = simple_noise_elimination(
            smoothed_signal, fs_dict['BVP'], std_ths
        )
        
        df = df.iloc[clean_signal_indices, :]
        df = df.reset_index(drop=True)
    
    grouped, baseline, stress, amusement = seperate_data_by_label(df)
    
    baseline_samples = get_samples(baseline, 0, ma_usage)
    stress_samples = get_samples(stress, 1, ma_usage)
    amusement_samples = get_samples(amusement, 0, ma_usage)
    
    print(f"stress: {len(stress_samples)}")
    print(f"non-stress: {len(amusement_samples) + len(baseline_samples)}")
    
    all_samples = pd.concat([baseline_samples, stress_samples, amusement_samples])
    all_samples = pd.concat([
        all_samples.drop('label', axis=1), 
        pd.get_dummies(all_samples['label'])
    ], axis=1)
    
    all_samples.to_csv(f'{savePath}{subject_feature_path}/S{subject_id}.csv')
    
    return len(all_samples)

#%% Main Processing
if __name__ == "__main__":
    total_window_len = 0
    BP, FREQ, TIME = False, False, False
    subject_ids = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    
    feat_names = None
    savePath = '../../data/WESAD'
    
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    
    for n in NOISE:
        if 'bp' in n.split('_'):
            BP = True
        if 'time' in n.split('_'):
            TIME = True
        
        subject_feature_path = '/subject_final_' + n + str(WINDOW_IN_SECONDS)
        merged_path = '/wesad_features_merged_' + n + '.csv'
        
        if not os.path.exists(savePath + subject_feature_path):
            os.makedirs(savePath + subject_feature_path)
        
        start_time = time.time()
        for patient in subject_ids:
            print(f'Processing data for S{patient}...')
            window_len = make_patient_data(patient, BP)
            total_window_len += window_len
        end_time = time.time()
        
        print(f"Processing time for patients: {end_time - start_time} seconds")
        
        combine_files(subject_ids)
        print(f'total_Window_len: {total_window_len}')
        print(f'Processing complete. {n}')
        total_window_len = 0