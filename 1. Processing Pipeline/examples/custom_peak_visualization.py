import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from processing_steps.processing_funcs import bandpass_filter, moving_average_filter, eliminate_noise_in_time, statistic_threshold, standardize, simple_bandpassfilter, simple_noise_elimination, simple_dynamic_threshold
from peak_detection.peak_detection_funcs import (
    threshold_peakdetection,
    first_derivative_with_adaptive_ths,
    moving_averages_with_dynamic_ths,
    lmm_peakdetection,
    ensemble_peak,
)



def visualize_separate_peaks(raw_ppg):

    start_index = 500
    delayed_ppg = raw_ppg[start_index:]
    
    # Step 2: Apply each filter in sequence
    fs = 50 #I think this is 50 for the earlier data!!
    cycle = 5

    # Apply bandpass filter
    bp_ppg = bandpass_filter(delayed_ppg, 0.7, 5, fs)
    # Apply moving average filter
    fwd = moving_average_filter(bp_ppg, window_size=3)
    bwd = moving_average_filter(bp_ppg[::-1], window_size=3)
    smoothed_ppg = np.mean(np.vstack((fwd, bwd[::-1])), axis=0)
    #normalize data
    norm_PPG = standardize(smoothed_ppg)
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

    # Run each peak detection function
    peaks_threshold = threshold_peakdetection(final_smoothed_ppg, fs)
    peaks_first_derivative = first_derivative_with_adaptive_ths(final_smoothed_ppg, fs)
    peaks_moving_avg = moving_averages_with_dynamic_ths(final_smoothed_ppg, sampling_rate=fs)
    peaks_lmm = lmm_peakdetection(final_smoothed_ppg, fs)
    peaks_ensemble = ensemble_peak(final_smoothed_ppg, fs)

    # Visualization: Separate graphs for each method
    methods = [
        ("Threshold Detection", peaks_threshold),
        ("First Derivative", peaks_first_derivative),
        ("Moving Average", peaks_moving_avg),
        ("LMM Detection", peaks_lmm),
        ("Ensemble Peaks", peaks_ensemble),
    ]

    plt.figure(figsize=(15, len(methods) * 4))

    for i, (method_name, peaks) in enumerate(methods, 1):
        plt.subplot(len(methods), 1, i)
        plt.plot(final_smoothed_ppg, color="blue", alpha=0.7, label="Filtered PPG Signal")
        plt.scatter(
            peaks, final_smoothed_ppg[peaks], color="red", label=f"{method_name} Peaks", s=20, marker="x"
        )
        plt.title(f"Peak Detection: {method_name}")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid()

    plt.tight_layout()
    plt.show()


def visualize_simple_peakdetect(raw_ppg):

    start_index = 500
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
    norm_PPG = standardize(smoothed_ppg)

    # Apply denoising filter
    segment_stds, std_ths = simple_dynamic_threshold(norm_PPG, 95)
    noise_eliminated_clean_indices = simple_noise_elimination(norm_PPG, fs, std_ths, window_size=3)
    
    # if len(noise_eliminated_clean_indices) == 0:
    #     print("No clean data detected. Adjust thresholds or inspect the input signal.")
    #     return {}
    # cleaned_ppg = norm_PPG[noise_eliminated_clean_indices]
    # # Apply a second smoothing filter
    # final_smoothed_ppg = moving_average_filter(cleaned_ppg, window_size=3)
    final_smoothed_ppg = norm_PPG

    # Run each peak detection function
    peaks_threshold = threshold_peakdetection(final_smoothed_ppg, fs)

    # Plot each step for comparison
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
    plt.scatter(
            peaks_threshold, final_smoothed_ppg[peaks_threshold], color="red", label=f"Peaks", s=20, marker="x"
        )
    plt.title("Final Smoothed and Denoised PPG Data")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()

# Run the visualization function
recording_csv = '../data/raw_data/selected_ppg_data.csv'
data = pd.read_csv(recording_csv)
ppg_signal = data["0FF74BB2-13DC-4453-BDBA-3694B9DB1BE5"].dropna().values
visualize_simple_peakdetect(ppg_signal)
