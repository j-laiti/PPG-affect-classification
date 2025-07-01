# This file outlines the functions for processing WESAD dataset
# These functions were adapted from the original code by Heo et al. (2021)
# https://doi.org/10.1109/ACCESS.2021.3060441

#%%
print("warm up")

#%% imports
# +
import os
import pickle
import math
import numpy as np
import pandas as pd

#%%

from scipy.interpolate import UnivariateSpline
from preprocessing.feature_extraction import *
from preprocessing.filters import *

# +
WINDOW_IN_SECONDS = 120


# If you want to apply noise filtering(band-pass filter) and noise elimination, include 'bp' and 'time' each in variable NOISE.
NOISE = ['bp_time']
main_path='../../data/WESAD/'


# +
# E4 (wrist) Sampling Frequencies

fs_dict = {'ACC': 32, 'BVP': 64, 'EDA': 4, 'TEMP': 4, 'label': 700, 'Resp': 700}

label_dict = {'baseline': 0, 'stress': 1, 'amusement': 0}
int_to_label = {0: 'baseline', 1: 'stress', 0: 'amusement'}
    
sec = 12
N = fs_dict['BVP']*sec  # one block : 10 sec
overlap = int(np.round(N * 0.02)) # overlapping length
overlap = overlap if overlap%2 ==0 else overlap+1


#%%

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



def get_samples(data, label, ma_usage):
    global feat_names
    global WINDOW_IN_SECONDS

    samples = []

    window_len = fs_dict['BVP'] * WINDOW_IN_SECONDS  # 64*60 , sliding window: 0.25 sec (60*0.25 = 15)   
    sliding_window_len = int(fs_dict['BVP'] * WINDOW_IN_SECONDS * 0.25)
    
    winNum = 0
    method = True
    
    i = 0
    while sliding_window_len * i <= len(data) - window_len:
        
         # 한 윈도우에 해당하는 모든 윈도우 담기,
        w = data[sliding_window_len * i: (sliding_window_len * i) + window_len]  
        # Calculate stats for window
        wstats = get_ppg_features(ppg_seg=w['BVP'].tolist(), fs = fs_dict['BVP'] , label=label, calc_sq=False)
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


def make_patient_data(subject_id, ma_usage):
    global savePath
    global WINDOW_IN_SECONDS
    
    # Make subject data object for Sx
    subject = SubjectData(main_path=main_path, subject_number=subject_id)
    
    # Empatica E4 data
    e4_data_dict = subject.get_wrist_data()

    # norm type
    norm_type = 'std'

    # fetch data and standardize it
    df = extract_ppg_data(e4_data_dict, subject.labels, norm_type)
    df_BVP = df.BVP
    df_BVP = df_BVP.tolist()


    #signal preprocessing 

    bp_bvp = bandpass_filter(df_BVP, 0.2, 10, fs_dict['BVP'], order=2)
    
    if BP:   
        df['BVP'] = bp_bvp
        
    if TIME:
        smoothed_signal = moving_average_filter(bp_bvp, window_size=5)
        # bwd = moving_average(bp_bvp[::-1], size=3)
        # bp_bvp = np.mean(np.vstack((fwd,bwd[::-1])), axis=0)
        df['BVP'] = smoothed_signal

        segment_stds, std_ths = simple_dynamic_threshold(smoothed_signal, fs_dict['BVP'], 85, window_size= 3)

        # Print the dynamic threshold and standard deviations
        print(f"Dynamic Threshold (95th Percentile): {std_ths}")
        print(f"Segment Standard Deviations: {segment_stds}")

        sim_clean_signal, clean_signal_indices = simple_noise_elimination(smoothed_signal, fs_dict['BVP'], std_ths)
        # sim_final_clean_signal = moving_average_filter(sim_clean_signal, window_size=3)
        
        df = df.iloc[clean_signal_indices,:]
        df = df.reset_index(drop=True)

        # df['BVP'] = sim_final_clean_signal
        
        # plt.figure(figsize=(40,20))
        # plt.plot(df['BVP'][:6000], color = 'b', linewidth=2.5)
    
    
    grouped, baseline, stress, amusement = seperate_data_by_label(df)   
    
    
    baseline_samples = get_samples(baseline, 0, ma_usage)
    stress_samples = get_samples(stress, 1, ma_usage)
    amusement_samples = get_samples(amusement, 0, ma_usage)
    
    print("stress: ",len(stress_samples))
    print("non-stress: ",len(amusement_samples)+len(baseline_samples))
    window_len = len(baseline_samples)+len(stress_samples)+len(amusement_samples)

    all_samples = pd.concat([baseline_samples, stress_samples, amusement_samples])
    all_samples = pd.concat([all_samples.drop('label', axis=1), pd.get_dummies(all_samples['label'])], axis=1) # get dummies로 원핫벡터로 라벨값 나타냄
    
    
    all_samples.to_csv(f'{savePath}{subject_feature_path}/S{subject_id}.csv')

    # Does this save any space?
    subject = None
    
    return window_len

# +
noise = NOISE[0].split('_')[:-1]
name = ''
for i, n in enumerate(noise):
    name += n
    if i != len(noise)-1:
        name += '_'
    
print(name)

# +
total_window_len = 0
BP, FREQ, TIME = False, False, False
subject_ids = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
# subject_ids = [7, 8, 9, 10, 11, 13, 14, 15, 16, 17]

feat_names = None
savePath = '../../data/WESAD'

if not os.path.exists(savePath):
    os.makedirs(savePath)


for n in NOISE:
    if 'bp' in n.split('_'):
        BP = True
    if 'time' in n.split('_'):
        TIME = True


    subject_feature_path = '/subject_bp02_extracted_features_' + n + str(WINDOW_IN_SECONDS)
    merged_path = '/data_bp8_merged_' + n +'.csv'
    
    if not os.path.exists(savePath + subject_feature_path):
        os.makedirs(savePath + subject_feature_path)
    
        
    for patient in subject_ids:
        print(f'Processing data for S{patient}...')
        window_len = make_patient_data(patient, BP)
        total_window_len += window_len

    combine_files(subject_ids)
    print('total_Window_len: ',total_window_len)
    print('Processing complete.', n)
    total_window_len = 0
