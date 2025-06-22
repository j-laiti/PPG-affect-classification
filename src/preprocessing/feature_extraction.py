# feature_extraction_JL.py
import numpy as np
import math
from scipy.interpolate import UnivariateSpline
import nolds
from scipy import stats
import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.processing.peak_detection_funcs import threshold_peakdetection, ensemble_peak

def calc_RRI(peaklist, fs):
    if len(peaklist) < 2:
        return [], [], []

    RR_list = [(peaklist[i + 1] - peaklist[i]) / fs * 1000.0 for i in range(len(peaklist) - 1)]
    
    # Remove RR intervals outside physiological range (300–2000 ms)
    RR_list = [rr for rr in RR_list if 300 <= rr <= 2000]

    if not RR_list:  # Handle case where all intervals are filtered out
        return [], [], []

    mean_RR = np.mean(RR_list)
    std_RR = np.std(RR_list)
    lower_bound, upper_bound = mean_RR - (1.5 * std_RR), mean_RR + (1.5 * std_RR)

    # Filter out outliers
    RR_list_e = [rr for rr in RR_list if lower_bound <= rr <= upper_bound]

    # Compute differences and squared differences
    RR_diff = np.abs(np.diff(RR_list_e))
    RR_sqdiff = np.diff(RR_list_e) ** 2

    return RR_list_e, RR_diff, RR_sqdiff


def calc_heartrate(RR_list):
    HR = []
    window_size = 10

    for val in RR_list:
        if val > 400 and val < 1500:
            heart_rate = 60000.0 / val
        elif (val > 0 and val < 400) or val > 1500:
            if len(HR) > 0:
                heart_rate = np.mean(HR[-window_size:])
            else:
                heart_rate = 60.0
        else:
            heart_rate = 0.0
        HR.append(heart_rate)

    return HR

def calc_td_hrv(RR_list, RR_diff, RR_sqdiff):
    HR = calc_heartrate(RR_list)
    HR_mean, HR_std = np.mean(HR), np.std(HR)
    meanNN, SDNN, medianNN = np.mean(RR_list), np.std(RR_list), np.median(np.abs(RR_list))
    meanSD, SDSD = np.mean(RR_diff) , np.std(RR_diff)
    RMSSD = np.sqrt(np.mean(RR_sqdiff))
    NN20 = [x for x in RR_diff if x > 20]
    NN50 = [x for x in RR_diff if x > 50]
    pNN20 = len(NN20) / len(RR_diff) * 100
    pNN50 = len(NN50) / len(RR_diff) * 100
    bar_y, bar_x = np.histogram(RR_list)
    TINN = np.max(bar_x) - np.min(bar_x)

    features = {'HR_mean': HR_mean, 'HR_std': HR_std, 'meanNN': meanNN, 'SDNN': SDNN, 'medianNN': medianNN,
                'meanSD': meanSD, 'SDSD': SDSD, 'RMSSD': RMSSD, 'pNN20': pNN20, 'pNN50': pNN50, 'TINN': TINN}

    return features

# frequency domain features not used in this code
def calc_fd_hrv(RR_list):
    rr_x = []
    pointer = 0
    for x in RR_list:
        pointer += x
        rr_x.append(pointer)
        
    if len(rr_x) <= 3 or len(RR_list) <= 3:
        return 0
    
    RR_x_new = np.linspace(rr_x[0], rr_x[-1], int(rr_x[-1]))
    interpolated_func = UnivariateSpline(rr_x, RR_list, k=3)
    datalen = len(RR_x_new)
    frq = np.fft.fftfreq(datalen, d=((1/1000.0)))
    frq = frq[range(int(datalen/2))]
    Y = np.fft.fft(interpolated_func(RR_x_new))/datalen
    Y = Y[range(int(datalen/2))]
    psd = np.power(Y, 2)

    lf = np.trapz(abs(psd[(frq >= 0.04) & (frq <= 0.15)]))
    hf = np.trapz(abs(psd[(frq > 0.15) & (frq <= 0.5)]))
    ulf = np.trapz(abs(psd[frq < 0.003]))
    vlf = np.trapz(abs(psd[(frq >= 0.003) & (frq < 0.04)]))
    
    if hf != 0:
        lfhf = lf/hf
    else:
        lfhf = 0
        
    total_power = lf + hf + vlf
    lfp = lf / total_power
    hfp = hf / total_power

    features = {'LF': lf, 'HF': hf, 'ULF' : ulf, 'VLF': vlf, 'LFHF': lfhf, 'total_power': total_power, 'lfp': lfp, 'hfp': hfp}
    return features

# non linear features not used in this code
def calc_nonli_hrv(RR_list):
    diff_RR = np.diff(RR_list)
    sd_heart_period = np.std(diff_RR, ddof=1) ** 2
    SD1 = np.sqrt(sd_heart_period * 0.5)
    SD2 = 2 * sd_heart_period - 0.5 * sd_heart_period
    pA = SD1*SD2
    
    if SD2 != 0:
        pQ = SD1 / SD2
    else:
        pQ = 0
    
    ApEn = nolds.sampen(RR_list)
    shanEn = stats.entropy(RR_list)
    D2 = nolds.corr_dim(RR_list, emb_dim=2)

    features = {'SD1': SD1, 'SD2': SD2, 'pA': pA, 'pQ': pQ, 'ApEn': ApEn, 'shanEn': shanEn, 'D2': D2}
    return features

def get_ppg_features(ppg_seg, fs, label, raw_ppg_signal=0, calc_sq=False):

    if len(ppg_seg) < fs * 10:  # Ensure at least 10 seconds of data
        print(f"Less than 10 sec of data")
        return {}

    peak = threshold_peakdetection(ppg_seg, fs)

    RR_list, RR_diff, RR_sqdiff = calc_RRI(peak, fs)
    print(f"RR intervals before filtering: {RR_list}")
    if len(RR_list) <= 3:
        return []
    
    td_features = calc_td_hrv(RR_list, RR_diff, RR_sqdiff)
    # fd_features = calc_fd_hrv(RR_list)
    
    # if fd_features == 0:
    #     return []
    
    # nonli_features = calc_nonli_hrv(RR_list)
    
    total_features = {**td_features}
    total_features["label"] = label

    # Calculate signal quality
    if calc_sq:
        sqi, snr, snr_freq = calculate_signal_quality(raw_ppg_signal, peak, fs=fs)

        # Add SQI and SNR to features
        total_features["sqi"] = sqi
        total_features["snr"] = snr
        total_features["snr_freq"] = snr_freq

    return total_features


def calculate_signal_quality(raw_ppg_signal, peaklist, fs=50):
    """
    Estimate signal quality based on absolute SNR, RR interval plausibility, and rr_std.

    Parameters:
    - ppg_seg (array): Filtered PPG signal segment.
    - peaklist (list): Detected peaks in the signal.
    - fs (int): Sampling frequency of the PPG signal.

    Returns:
    - sqi (float): Signal quality index (0 to 1, higher is better).
    - quality_flags (dict): Detailed signal quality indicators.
    """
    # 1. Absolute SNR
    snr = np.abs(np.std(raw_ppg_signal) / np.mean(raw_ppg_signal)) if np.mean(raw_ppg_signal) != 0 else 0

    # 2. RR Intervals and Plausibility
    if len(peaklist) < 2:
        rr_mean, rr_std = 0, 0
    else:
        rr_intervals = [(peaklist[i + 1] - peaklist[i]) / fs * 1000 for i in range(len(peaklist) - 1)]
        rr_mean = np.mean(rr_intervals)
        rr_std = np.std(rr_intervals)

    # 3. Combine into SQI
    rr_std_clipped = min(rr_std, 1000)  # Cap rr_std to avoid excessive influence
    rr_mean_penalty = 1 if rr_mean < 300 or rr_mean > 2000 else 0  # Penalty for implausible mean RR

    # SQI computation
    sqi = max(0, 1 - (0.4 * (1 - min(snr / 100, 1)) + 0.4 * (rr_std_clipped / 1000) + 0.2 * rr_mean_penalty))

    # snr based on frequency
    # perform a lowpass filter for signal under 20 Hz
    ppg_signal = lowpass_filter(raw_ppg_signal, 20, 50)

    # perform a highpass filter for signal above 20 Hz
    noise_signal = highpass_filter(raw_ppg_signal, 20, 50)

    # find the amplitude of each and then make a ratio
    signal_amplitude = np.sqrt(np.mean(ppg_signal ** 2))  # RMS amplitude
    noise_amplitude = np.sqrt(np.mean(noise_signal ** 2))  # RMS amplitude
    snr_freq = 20 * np.log10(signal_amplitude / noise_amplitude)  # Amplitude-based SNR

    return sqi, snr, snr_freq

from scipy.signal import butter, filtfilt

def lowpass_filter(data, cutoff, fs, order=2):
    """
    Apply a low-pass Butterworth filter to the signal.

    Parameters:
    - data: The input signal.
    - cutoff: The cutoff frequency in Hz.
    - fs: The sampling frequency of the signal.
    - order: The order of the filter (default is 4).

    Returns:
    - Filtered signal.
    """
    nyquist = 0.5 * fs  # Nyquist frequency
    normalized_cutoff = cutoff / nyquist  # Normalize the cutoff frequency
    b, a = butter(order, normalized_cutoff, btype='low', analog=False)  # Design the filter
    filtered_data = filtfilt(b, a, data)  # Apply the filter
    return filtered_data

def highpass_filter(data, cutoff, fs, order=2):
    """
    Apply a high-pass Butterworth filter to the signal.

    Parameters:
    - data: The input signal.
    - cutoff: The cutoff frequency in Hz.
    - fs: The sampling frequency of the signal.
    - order: The order of the filter (default is 4).

    Returns:
    - Filtered signal.
    """
    nyquist = 0.5 * fs  # Nyquist frequency
    normalized_cutoff = cutoff / nyquist  # Normalize the cutoff frequency
    b, a = butter(order, normalized_cutoff, btype='high', analog=False)  # Design the filter
    filtered_data = filtfilt(b, a, data)  # Apply the filter
    return filtered_data

