# This file outlines the code to process PPG data, calling filtering, peak detection, and feature extraction methods
# The first section outlines the steps to import and process custom raw PPG data 
# The second section defines variables specific to the WESAD dataset, extracting PPG data from the dataset, and saving the features

# library imports
import os
from utils.processing.wesad_processing import *

# window length for feature extraction
window_in_seconds = 120

# path for WESAD dataset
main_path='data/WESAD/'

# Sampling Frequencies
ppg_fs = 64 # Empatica E4 BVP Frequency
label_fs = 700 # Label Frequency

# labels for WESAD dataset
int_to_label = {0: 'baseline', 1: 'stress', 0: 'amusement'}

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
    window_len = make_patient_data(patient, ppg_fs, label_fs, main_path, savePath, subject_feature_path, window_in_seconds)


combine_files(subject_ids, savePath, subject_feature_path, merged_path)

print('Processing complete.')

