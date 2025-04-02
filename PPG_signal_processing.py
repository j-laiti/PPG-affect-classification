# -*- coding: utf-8 -*-
# +
import os
import pickle
import math
import numpy as np
import pandas as pd

from scipy.interpolate import UnivariateSpline
from utils.processing.processing_funcs import *

# Feature extraction of WESAD dataset besed on Heo et.al. (2021)
# https://doi.org/10.1109/ACCESS.2021.3060441

# window length for feature extraction
WINDOW_IN_SECONDS = 120

# path for WESAD dataset
main_path='data/WESAD/'

# empatica E4 Sampling Frequencies
fs_dict = {'BVP': 64, 'label': 700}

# labels for WESAD dataset
label_dict = {'baseline': 0, 'stress': 1, 'amusement': 0}
int_to_label = {0: 'baseline', 1: 'stress', 0: 'amusement'}

# TODO: can I delete this? 
sec = 10
N = fs_dict['BVP']*sec  # one block : 10 sec
overlap = int(np.round(N * 0.02)) # overlapping length
overlap = overlap if overlap%2 ==0 else overlap+1


# 
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


def extract_ppg_data(e4_data_dict, labels, norm_type=None):
    # Dataframes for each sensor type
    df = pd.DataFrame(e4_data_dict['BVP'], columns=['BVP'])
    label_df = pd.DataFrame(labels, columns=['label'])
    

    # Adding indices for combination due to differing sampling frequencies
    df.index = [(1 / fs_dict['BVP']) * i for i in range(len(df))]
    label_df.index = [(1 / fs_dict['label']) * i for i in range(len(label_df))]

    # Change indices to datetime
    df.index = pd.to_datetime(df.index, unit='s')
    label_df.index = pd.to_datetime(label_df.index, unit='s')

    df = df.join(label_df, how='outer')
    
    df['label'] = df['label'].bfill()
    
    df.reset_index(drop=True, inplace=True)
    
    if norm_type == 'std':  # 시그널 자체를 normalization
        # std norm
        df['BVP'] = (df['BVP'] - df['BVP'].mean()) / df['BVP'].std()
    elif norm_type == 'minmax':
        # minmax norm
        df = (df - df.min()) / (df.max() - df.min())

    # Groupby
    df = df.dropna(axis=0) # nan인 행 제거
    
    return df


def seperate_data_by_label(df):
    
    grouped = df.groupby('label')
    baseline = grouped.get_group(1)
    stress = grouped.get_group(2)
    amusement = grouped.get_group(3)   
    
    return grouped, baseline, stress, amusement



def get_samples(data, label):
    global WINDOW_IN_SECONDS

    samples = []

    window_len = fs_dict['BVP'] * WINDOW_IN_SECONDS  # 64*60 , sliding window: 0.25 sec (60*0.25 = 15)   
    sliding_window_len = int(fs_dict['BVP'] * WINDOW_IN_SECONDS * 0.25)
    
    winNum = 0
    
    i = 0
    while sliding_window_len * i <= len(data) - window_len:
        
         # 
        w = data[sliding_window_len * i: (sliding_window_len * i) + window_len]  
        # Calculate stats for window
        wstats = get_ppg_features(ppg_seg=w['BVP'].tolist(), window_length = window_len, label=label)
        winNum += 1
        
        if wstats == []:
            i += 1
            continue;
        # Seperating sample and label
        x = pd.DataFrame(wstats, index = [i])
    
        samples.append(x)
        i += 1

    return pd.concat(samples)


def combine_files(subjects):
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
    print('Number of samples per class:')
    for label, number in zip(counts.index, counts.values):
        print(f'{int_to_label[label]}: {number}')


def make_patient_data(subject_id):
    global savePath
    global WINDOW_IN_SECONDS
    
    # Make subject data object for Sx
    subject = SubjectData(main_path=main_path, subject_number=subject_id)
    
    # Empatica E4 data
    e4_data_dict = subject.get_wrist_data()

    # norm type
    norm_type = 'std'

    df = extract_ppg_data(e4_data_dict, subject.labels, norm_type)
    df_BVP = df.BVP
    df_BVP = df_BVP.tolist()


    #signal preprocessing 

    # bp_bvp = simple_bandpassfilter(b, a, df_BVP)
    bp_bvp = butter_bandpassfilter(df_BVP, 0.5, 10, fs_dict['BVP'], order=2)
    
    df['BVP'] = bp_bvp

    smoothed_signal = moving_average_filter(bp_bvp, window_size=5)
    # bwd = moving_average(bp_bvp[::-1], size=3)
    # bp_bvp = np.mean(np.vstack((fwd,bwd[::-1])), axis=0)
    df['BVP'] = smoothed_signal

    segment_stds, std_ths = simple_dynamic_threshold(smoothed_signal, 85, window_size=fs_dict['BVP'] * 3)

    # Print the threshold and standard deviations
    print(f"Dynamic Threshold (85th Percentile): {std_ths}")
    print(f"Segment Standard Deviations: {segment_stds}")

    sim_clean_signal, clean_signal_indices = simple_noise_elimination(smoothed_signal, fs_dict['BVP'],std_ths)
    # sim_final_clean_signal = moving_average_filter(sim_clean_signal, window_size=3)
    
    df = df.iloc[clean_signal_indices,:]
    df = df.reset_index(drop=True)
    
    
    group, baseline, stress, amusement = seperate_data_by_label(df)   
    
    
    baseline_samples = get_samples(baseline, 0)
    stress_samples = get_samples(stress, 1)
    amusement_samples = get_samples(amusement, 0)
    
    print("stress: ",len(stress_samples))
    print("non-stress: ",len(amusement_samples)+len(baseline_samples))
    window_len = len(baseline_samples)+len(stress_samples)+len(amusement_samples)

    all_samples = pd.concat([baseline_samples, stress_samples, amusement_samples])
    all_samples = pd.concat([all_samples.drop('label', axis=1), pd.get_dummies(all_samples['label'])], axis=1) # get dummies
    
    
    all_samples.to_csv(f'{savePath}{subject_feature_path}/S{subject_id}.csv')

    # Does this save any space?
    subject = None
    
    return window_len

# +
total_window_len = 0
subject_ids = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]

savePath = 'data'

if not os.path.exists(savePath):
    os.makedirs(savePath)

subject_feature_path = '/WESAD_subject_features'
merged_path = '/WESAD_data_merged.csv'

if not os.path.exists(savePath + subject_feature_path):
    os.makedirs(savePath + subject_feature_path)
    
for patient in subject_ids:
    print(f'Processing data for S{patient}...')
    window_len = make_patient_data(patient)
    total_window_len += window_len

combine_files(subject_ids)
print('total_Window_len: ',total_window_len)
print('Processing complete.', n)
total_window_len = 0

