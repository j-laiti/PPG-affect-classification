import numpy as np
from scipy.signal import find_peaks
from preprocessing.filters import moving_average_filter

#%% Peak detection functions

def threshold_peakdetection(dataset, fs):
    dataset = dataset.copy()  # Ensure the input signal is not modified
    window = []
    peaklist = []
    localaverage = np.average(dataset)
    TH_elapsed = np.ceil(0.45 * fs)  # Threshold for elapsed time between peaks in samples
    
    # Step 1: Identify all candidate peaks
    for listpos, datapoint in enumerate(dataset):
        if datapoint < localaverage and len(window) < 1:
            continue
        elif datapoint >= localaverage:
            window.append(datapoint)
        else:
            maximum = max(window)
            beatposition = listpos - len(window) + window.index(maximum)
            peaklist.append(beatposition)
            window = []

    # Step 2: Filter peaks based on TH_elapsed
    peakarray = []
    for val in peaklist:
        if len(peakarray) > 0:  # Check if there are any peaks in peakarray
            prev_peak = peakarray[-1]  # Use the last filtered peak
            elapsed = val - prev_peak
            if elapsed > TH_elapsed:
                peakarray.append(val)
        else:
            peakarray.append(val)  # Always add the first peak
    
    return peakarray  # Return the filtered peak array


def determine_peak_or_not(prevAmp, curAmp, nextAmp):
    return prevAmp < curAmp and curAmp >= nextAmp

def onoff_set(peak, sig):
    onoffset = []
    for p in peak:
        onset = offset = None
        for i in range(p, 0, -1):
            if sig[i] == 0:
                onset = i
                break
        for j in range(p, len(sig)):
            if sig[j] == 0:
                offset = j
                break
        if onset is not None and offset is not None and onset < offset:
            onoffset.append([onset, offset])
    return onoffset


def first_derivative_with_adaptive_ths(data, fs):
    data = data.copy()  # Ensure the input signal is not modified
    peak = []
    divisionSet = []
    for divisionUnit in range(0, len(data) - 1, 5 * fs):
        eachDivision = data[divisionUnit: (divisionUnit + 1) * 5 * fs]
        divisionSet.append(eachDivision)
    
    selectiveWindow = 2 * fs
    block_size = 5 * fs
    bef_idx = -300
    
    for divInd in range(len(divisionSet)):
        block = divisionSet[divInd]
        ths = np.mean(block[:selectiveWindow])
        firstDeriv = block[1:] - block[:-1]
        for i in range(1, len(firstDeriv)):
            if firstDeriv[i] <= 0 and firstDeriv[i - 1] > 0:
                if block[i] > ths:
                    idx = block_size * divInd + i
                    if idx - bef_idx > (300 * fs / 1000):
                        peak.append(idx)
                        bef_idx = idx
    return peak

def moving_averages_with_dynamic_ths(input_signal, sampling_rate=50, peakwindow=.111, beatwindow=.667, beatoffset=.02, mindelay=.3, show=False):
    # Copy the input signal to avoid modifying the original
    signal_copy = input_signal.copy()
    signal_copy[signal_copy < 0] = 0
    sqrd = signal_copy ** 2

    ma_peak_kernel = int(np.rint(peakwindow * sampling_rate))
    ma_peak = moving_average_filter(sqrd, ma_peak_kernel)

    ma_beat_kernel = int(np.rint(beatwindow * sampling_rate))
    ma_beat = moving_average_filter(sqrd, ma_beat_kernel)

    thr1 = ma_beat + beatoffset * np.mean(sqrd)
    waves = ma_peak > thr1
    beg_waves = np.where(np.logical_and(np.logical_not(waves[:-1]), waves[1:]))[0]
    end_waves = np.where(np.logical_and(waves[:-1], np.logical_not(waves[1:])))[0]
    end_waves = end_waves[end_waves > beg_waves[0]]

    num_waves = min(beg_waves.size, end_waves.size)
    min_len = int(np.rint(peakwindow * sampling_rate))
    min_delay = int(np.rint(mindelay * sampling_rate))
    peaks = [0]

    for i in range(num_waves):
        beg = beg_waves[i]
        end = end_waves[i]
        len_wave = end - beg
        if len_wave < min_len:
            continue

        data = signal_copy[beg:end]
        locmax, props = find_peaks(data, prominence=(None, None))
        if locmax.size > 0:
            peak = beg + locmax[np.argmax(props["prominences"])]
            if peak - peaks[-1] > min_delay:
                peaks.append(peak)

    peaks.pop(0)
    return [int(p) for p in peaks]


def lmm_peakdetection(data, fs):
    data = data.copy()  # Ensure the input signal is not modified
    peaks, _ = find_peaks(data, height=0)
    return [int(peak) for peak in peaks if data[peak] > 0]

def ensemble_peak(preprocessed_data, fs, ensemble_ths=3):
    preprocessed_data = preprocessed_data.copy()  # Ensure the input signal is not modified
    peak1 = threshold_peakdetection(preprocessed_data, fs)
    peak2 = first_derivative_with_adaptive_ths(preprocessed_data, fs)
    peak3 = moving_averages_with_dynamic_ths(preprocessed_data, sampling_rate=fs)
    peak4 = lmm_peakdetection(preprocessed_data, fs)

    all_peaks = peak1 + peak2 + peak3 + peak4

    peak_dic = {}
    for key in all_peaks:
        if key in peak_dic:
            peak_dic[key] += 1
        else:
            peak_dic[key] = 1

    final_peak = [key for key, value in peak_dic.items() if value >= ensemble_ths]
    return final_peak