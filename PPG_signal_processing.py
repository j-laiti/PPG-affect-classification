# This file outlines the code to process PPG data, calling filtering, peak detection, and feature extraction methods

# The first section outlines the steps to import and process custom raw PPG data 
# The second section defines variables specific to the WESAD dataset, extracting PPG data from the dataset, and saving the features

#%% library imports
import os
from utils.processing.wesad_processing import *

#%% WESAD dataset processing

# variables needed for processing
main_path='data/WESAD/' # path for WESAD dataset (change to your own path)
window_in_seconds = 120 # window length for feature extraction
ppg_fs = 64 # Empatica E4 BVP sampling frequency
label_fs = 700 # Label frequency
int_to_label = {0: 'baseline', 1: 'stress', 0: 'amusement'} # labels for WESAD dataset
subject_ids = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17] # subject IDs for WESAD dataset
savePath = 'data' # path to save the processed data (change to your own path)
subject_feature_path = '/WESAD_subject_features' # path to save the subject features
merged_path = '/WESAD_data_merged.csv' # path to save the merged features

# create the save paths if they don't exist
if not os.path.exists(savePath):
    os.makedirs(savePath)
if not os.path.exists(savePath + subject_feature_path):
    os.makedirs(savePath + subject_feature_path)

# process each PPG signal, extract features, and save to csv for each subject
for patient in subject_ids:
    print(f'Processing data for S{patient}...')
    window_len = make_patient_data(patient, ppg_fs, label_fs, main_path, savePath, subject_feature_path, window_in_seconds)

# combine the features for each subject
combine_files(subject_ids, savePath, subject_feature_path, merged_path)

print('Processing complete.')


# %% Custom PPG data processing
# The following code outlines the steps to import and process custom raw PPG data
# The import file format is columns of raw ppg signals with a heading (in this case the recording session ids) in a .csv format

ppg_data_csv = 'data/raw_data/selected_ppg_data.csv' # path to raw PPG data (change to find your own data)
ppg_data = pd.read_csv(ppg_data_csv)

features_list = []
columns = ['Session_ID']

for session_id in ppg_data.columns:
    print(f"Processing session: {session_id}")
    ppg_values = ppg_data[session_id].dropna().values
    
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
