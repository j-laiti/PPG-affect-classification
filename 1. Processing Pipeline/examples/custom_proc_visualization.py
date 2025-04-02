# visualize the processing process! hehe!
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from processing_funcs import bandpass_filter, moving_average_filter, eliminate_noise_in_time, statistic_threshold, normalize, simple_bandpassfilter, simple_noise_elimination, simple_statistic_threshold, simple_statistic_adaptive_threshold
# from fetchFirebasePPG import fetch_ppg_array

def visualize_processing_steps(raw_ppg):
    # Step 1: load csv
    # try:
    #     data = pd.read_csv(recording_csv)
    #     raw_ppg = data.iloc[:, 0].values  # Assuming PPG data is in the first column
    # except Exception as e:
    #     print(f"Error loading file {recording_csv}: {e}")
    #     return
    start_index = 500
    delayed_ppg = raw_ppg[start_index:]
    
    # Step 2: Apply each filter in sequence
    fs = 50 #I think this is 50 for the earlier data!!
    cycle = 5

    # Apply bandpass filter
    bp_ppg = bandpass_filter(delayed_ppg, 0.7, 5, fs)
    # bp_ppg = bp_ppg[50:]

    
    # Apply moving average filter
    fwd = moving_average_filter(bp_ppg, window_size=3)
    bwd = moving_average_filter(bp_ppg[::-1], window_size=3)
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
        return

    cleaned_ppg = smoothed_ppg[clean_indices]

    
    # Apply a second smoothing filter
    final_smoothed_ppg = moving_average_filter(cleaned_ppg, window_size=3)
    
    # Step 3: Plot each step for comparison
    plt.figure(figsize=(15, 8))

    plt.subplot(4, 1, 1)
    plt.plot(raw_ppg, color='blue', alpha=0.7)
    plt.title("Raw PPG Data")
    plt.ylabel("Amplitude")

    plt.subplot(4, 1, 2)
    plt.plot(bp_ppg, color='green', alpha=0.7)
    plt.title("Bandpass Filtered PPG Data")
    plt.ylabel("Amplitude")

    plt.subplot(4, 1, 3)
    plt.plot(norm_PPG, color='orange', alpha=0.7)
    plt.title("Normalized PPG Data")
    plt.ylabel("Amplitude")

    plt.subplot(4, 1, 4)
    plt.plot(final_smoothed_ppg, color='red', alpha=0.7)
    plt.title("Final Smoothed and Denoised PPG Data")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()


# simplified pipeline

def visualize_simple_processing_steps(raw_ppg):
    # Step 1: load csv
    # try:
    #     data = pd.read_csv(recording_csv)
    #     raw_ppg = data.iloc[:, 0].values  # Assuming PPG data is in the first column
    # except Exception as e:
    #     print(f"Error loading file {recording_csv}: {e}")
    #     return
    start_index = 1100
    delayed_ppg = raw_ppg[start_index:]
    
    # Step 2: Apply each filter in sequence
    fs = 50 #I think this is 50 for the earlier data!!

    # fs = 50, bp 0.7-5
    # b = [ 0.05223535, 0.0, -0.1044707, 0.0, 0.05223535]
    # a = [ 1.0, -3.16573522, 3.83025017, -2.12887887, 0.46652974]
    # fs = 50, bp 0.4-6
    b = [ 0.08149274, 0.0, -0.16298549, 0.0, 0.08149274]
    a = [ 1.0, -2.98616694, 3.37019878, -1.75526961, 0.37217724]

    # Apply bandpass filter
    bp_ppg = simple_bandpassfilter(b, a, delayed_ppg)

    # Apply moving average filter
    smoothed_ppg = moving_average_filter(bp_ppg, window_size=3)

    #normalize data
    norm_PPG = normalize(smoothed_ppg)

    # Apply denoising filter
    std_ths = simple_statistic_threshold(norm_PPG, 1.0)
    noise_eliminated_clean_indices = simple_noise_elimination(norm_PPG, fs, std_ths, window_size=3)
    
    if len(noise_eliminated_clean_indices) == 0:
        print("No clean data detected. Adjust thresholds or inspect the input signal.")
        return {}
    cleaned_ppg = norm_PPG[noise_eliminated_clean_indices]
    # Apply a second smoothing filter
    final_smoothed_ppg = moving_average_filter(cleaned_ppg, window_size=3)
    
    # Step 3: Plot each step for comparison
    plt.figure(figsize=(15, 8))

    plt.subplot(4, 1, 1)
    plt.plot(raw_ppg, color='blue', alpha=0.7)
    plt.title("Raw PPG Data")
    plt.ylabel("Amplitude")

    plt.subplot(4, 1, 2)
    plt.plot(bp_ppg, color='green', alpha=0.7)
    plt.title("Bandpass Filtered PPG Data")
    plt.ylabel("Amplitude")

    plt.subplot(4, 1, 3)
    plt.plot(norm_PPG, color='orange', alpha=0.7)
    plt.title("Normalized PPG Data")
    plt.ylabel("Amplitude")

    plt.subplot(4, 1, 4)
    plt.plot(final_smoothed_ppg, color='red', alpha=0.7)
    plt.title("Final Smoothed and Denoised PPG Data")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()

# Run the visualization function
recording_csv = '../data/raw_data/selected_ppg_data.csv'
data = pd.read_csv(recording_csv)
ppg_signal = data["47ee03bd-64c8-488e-96db-44fc40690766"].dropna().values
# call either a specific column header or a specific column number
visualize_simple_processing_steps(ppg_signal)