# This file outlines the functions for processing WESAD dataset
# These functions were adapted from the original code by Heo et al. (2021)
# https://doi.org/10.1109/ACCESS.2021.3060441

# imports
import os
import pickle
import math
import numpy as np
import pandas as pd

from scipy.interpolate import UnivariateSpline
from utils.processing.processing_funcs import *
from utils.processing.feature_extraction import *

# WESAD dataset class
class SubjectData:

    def __init__(self, main_path, subject_number):
        self.name = f'S{subject_number}'
        self.subject_keys = ['signal', 'label', 'subject']
        self.signal_keys = ['chest', 'wrist']
        self.chest_keys = ['ACC', 'ECG', 'EMG', 'EDA', 'Temp', 'Resp']
        self.wrist_keys = ['ACC', 'BVP', 'EDA', 'TEMP']
        with open(os.path.join(main_path, self.name) + '/' + self.name + '.pkl', 'rb') as file:
            self.data = pickle.load(file, encoding='latin1')
        self.labels = self.data['label']

    def get_wrist_data(self):
        data = self.data['signal']['wrist']
        data.update({'Resp': self.data['signal']['chest']['Resp']})
        return data
    
# PPG signal exctraction from WESAD
def extract_ppg_data(e4_data_dict, labels, ppg_fs, label_fs, norm_type=None):
    # Dataframes for each sensor type
    df = pd.DataFrame(e4_data_dict['BVP'], columns=['BVP'])
    label_df = pd.DataFrame(labels, columns=['label'])
    

    # Adding indices for combination due to differing sampling frequencies
    df.index = [(1 / ppg_fs) * i for i in range(len(df))]
    label_df.index = [(1 / label_fs) * i for i in range(len(label_df))]

    # Change indices to datetime
    df.index = pd.to_datetime(df.index, unit='s')
    label_df.index = pd.to_datetime(label_df.index, unit='s')

    df = df.join(label_df, how='outer')
    
    df['label'] = df['label'].bfill()
    
    df.reset_index(drop=True, inplace=True)
    
    if norm_type == 'std': # normalization
        # std norm
        df['BVP'] = (df['BVP'] - df['BVP'].mean()) / df['BVP'].std()
    elif norm_type == 'minmax':
        # minmax norm
        df = (df - df.min()) / (df.max() - df.min())

    # Groupby
    df = df.dropna(axis=0) # nan
    
    return df

# Seperate WESAD data by label
def seperate_data_by_label(df):
    
    grouped = df.groupby('label')
    baseline = grouped.get_group(1)
    stress = grouped.get_group(2)
    amusement = grouped.get_group(3)   
    
    return grouped, baseline, stress, amusement


# extract features from PPG signal for each label
def get_samples(data, label, ppg_fs, window_in_seconds):

    samples = []

    window_len = ppg_fs * window_in_seconds    
    sliding_window_len = int(ppg_fs * window_in_seconds * 0.25)
    
    winNum = 0
    
    i = 0
    while sliding_window_len * i <= len(data) - window_len:
        
         # 
        w = data[sliding_window_len * i: (sliding_window_len * i) + window_len]  
        # Calculate stats for window
        wstats = get_ppg_features(ppg_seg=w['BVP'].tolist(), fs = ppg_fs, label = label)
        winNum += 1
        
        if wstats == []:
            i += 1
            continue;
        # Seperating sample and label
        x = pd.DataFrame(wstats, index = [i])
    
        samples.append(x)
        i += 1

    return pd.concat(samples)

# combine all subjects' data into one dataframe and save csv
def combine_files(subjects, savePath, subject_feature_path, merged_path):
    df_list = []
    for s in subjects:
        df = pd.read_csv(f'{savePath}{subject_feature_path}/S{s}.csv', index_col=0)
        df['subject'] = s
        df_list.append(df)

    df = pd.concat(df_list)
    # Assume df['0'] and df['1'] are the one-hot encoded columns
    df['label'] = df[['0', '1']].idxmax(axis=1).astype(int)


    # df['label'] = (df['0'].astype(str) + df['1'].astype(str)).apply(lambda x: x.index('1'))  # Returns the index of the part that is 1
    df.drop(['0', '1'], axis=1, inplace=True)

    df.reset_index(drop=True, inplace=True)

    df.to_csv(savePath + merged_path)

    counts = df['label'].value_counts()
    # print('Number of samples per class:')
    # for label, number in zip(counts.index, counts.values):
    #     print(f'{int_to_label[label]}: {number}')


# process each PPG signal, extract features, and save to csv for each subject
def make_patient_data(subject_id, ppg_fs, label_fs, main_path, savePath, subject_feature_path, window_in_seconds):
    
    # Make subject data object for Sx
    subject = SubjectData(main_path=main_path, subject_number=subject_id)
    
    # Empatica E4 data
    e4_data_dict = subject.get_wrist_data()

    # norm type
    norm_type = 'std'

    df = extract_ppg_data(e4_data_dict, subject.labels, ppg_fs, label_fs, norm_type)
    df_BVP = df.BVP
    df_BVP = df_BVP.tolist()


    ## Signal processing ##

    # Bandpass filter
    bp_bvp = bandpass_filter(df_BVP, 0.5, 10, ppg_fs, order=2)
    df['BVP'] = bp_bvp

    # Moving average filter
    smoothed_signal = moving_average_filter(bp_bvp, window_size=5)
    df['BVP'] = smoothed_signal

    # Noise elimination
    segment_length = ppg_fs * 120  # 120 seconds
    clean_indices = []
    clean_signal = []
    threshold_percentile = 95

    for start in range(0, len(smoothed_signal), segment_length):
        segment = smoothed_signal[start:start + segment_length]

        # calculate the threshold for this segment
        std_ths = simple_dynamic_threshold(segment, threshold_percentile, window_size=ppg_fs * 3)

        # eliminate noise from the segment
        sim_clean_signal, clean_segment_indices = simple_noise_elimination(segment, ppg_fs, std_ths)

        # adjust the indices to match the original signal
        segment_clean_indices = [i + start for i in clean_segment_indices]

        # append the cleaned segment to the list
        clean_indices.extend(segment_clean_indices)
        clean_signal.extend(sim_clean_signal)

    # Filter the DataFrame to match the clean signal
    df = df.iloc[clean_indices, :]

    # Add the clean signal to the DataFrame
    df['clean_BVP'] = clean_signal

    # Reset the DataFrame index (optional, if needed for further processing)
    df = df.reset_index(drop=True)
    
    group, baseline, stress, amusement = seperate_data_by_label(df)   
    
    baseline_samples = get_samples(baseline, 0, ppg_fs, window_in_seconds)
    stress_samples = get_samples(stress, 1, ppg_fs, window_in_seconds)
    amusement_samples = get_samples(amusement, 0, ppg_fs, window_in_seconds)
    
    print("stress: ",len(stress_samples))
    print("non-stress: ",len(amusement_samples)+len(baseline_samples))
    window_len = len(baseline_samples)+len(stress_samples)+len(amusement_samples)

    all_samples = pd.concat([baseline_samples, stress_samples, amusement_samples])
    all_samples = pd.concat([all_samples.drop('label', axis=1), pd.get_dummies(all_samples['label'])], axis=1) # get dummies
    
    
    all_samples.to_csv(f'{savePath}{subject_feature_path}/S{subject_id}.csv')

    subject = None
    
    return window_len
