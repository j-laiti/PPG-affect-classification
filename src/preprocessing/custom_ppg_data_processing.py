# This function processes a column of raw PPG data using the processing, peak detection, and feature extraction functions

def process_raw_ppg_data(ppg_data_csv, ppg_fs, label_fs, window_in_seconds, save_path):
    # standardize the PPG signal

    # apply bandpass filter

    # smooth the signal with a moving average filter

    # remove noise from the signal (if signal is longer than 120 seconds, it will likely perform better if it is split into segments and process each segment separately)
    # calculate the dynamic threshold for noise elimination
    # eliminate noise from the signal

    # apply another moving average filter

    # run feature extraction function which includes threshold peak detection, RRI calculation, time-domain HRV feature extraction, and signal quality calculation

    return features_list
