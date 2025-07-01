# Processing functions for efficiency comparison to:
# S. Heo, S. Kwon and J. Lee, "Stress Detection With Single PPG Sensor by Orchestrating Multiple Denoising and Peak-Detecting Methods," in IEEE Access, vol. 9, pp. 47777-47785, 2021, 
# doi: 10.1109/ACCESS.2021.3060441.

### Filters ###
import math
from scipy import signal
from scipy.fft import fft, ifft
from scipy import fftpack
from scipy.stats import kurtosis, skew
from scipy.signal import butter, lfilter
from scipy import stats
from sklearn.linear_model import LinearRegression



import numpy as np

import matplotlib.pyplot as plt

# +
# filter


def heo_butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs  
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def heo_butter_bandpassfilter(data, lowcut, highcut, fs, order=5):
    b, a = heo_butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y



def heo_moving_average(data, periods=4):
    result = []
    data_set = np.asarray(data)
    weights = np.ones(periods) / periods
    result = np.convolve(data_set, weights, mode='valid')
    return result

def statistic_threshold(clean_signal, fs, ths):
    stds, kurtosiss, skews, valley = statistic_detection(clean_signal, fs)
    std_ths = np.mean(stds) + ths[0] # paper's threshold : 3.2  
    kurt_ths = np.mean(kurtosiss) + ths[1]  #3.1  
    skews_ths = [np.mean(skews) - ths[2], np.mean(skews) + ths[3]]  # -0.3 and 0.8   -0.5 , 0.9
    
    return std_ths, kurt_ths, skews_ths


def statistic_detection(signal, fs):
    
    valley = pair_valley(valley_detection(signal, fs))
    stds=[]
    kurtosiss=[]
    skews=[]

    for val in valley: # 사이클 한 번동안의 통계적 평균 리스트 저장
        stds.append(np.std(signal[val[0]:val[1]]))
        kurtosiss.append(kurtosis(signal[val[0]:val[1]]))
        skews.append(skew(signal[val[0]:val[1]])) 

    return stds, kurtosiss, skews, valley

def eliminate_noise_in_time(data, fs, ths,cycle=1):
    stds, kurtosiss,skews, valley = statistic_detection(data, fs)
    
    
    #cycle 수만큼 다시 평균내서 리스트 저장
    stds_, kurtosiss_, skews_ = [], [], []
    stds_ = [np.mean(stds[i:i+cycle]) for i in range(0,len(stds)-cycle+1,cycle)]
    kurtosiss_ = [np.mean(kurtosiss[i:i+cycle]) for i in range(0,len(kurtosiss)-cycle+1,cycle)]
    skews_ = [np.mean(skews[i:i+cycle]) for i in range(0,len(skews)-cycle+1,cycle)]    
   
    # extract clean index, 사이클 인덱스       
    eli_std = [stds_.index(x) for x in stds_ if x < ths[0]]
    eli_kurt = [kurtosiss_.index(x) for x in kurtosiss_ if x < ths[1]]
    eli_skew = [skews_.index(x) for x in skews_ if x > ths[2][0] and x < ths[2][1]]

    total_list = eli_std + eli_kurt + eli_skew
    
    
    # store the number of extracted each index(각 인덱스 extract된 횟수 저장)
    dic = dict()
    for i in total_list:
        if i in dic.keys():
            dic[i] += 1
        else:
            dic[i] = 1
            
    new_list = []
    for key, value in dic.items():
        if value >= 3:
            new_list.append(key)
    new_list.sort()
    
    eliminated_data = []
    index = []
    for x in new_list:
        index.extend([x for x in range(valley[x*cycle][0],valley[x*cycle+cycle-1][1],1)])

    print(len(data), len(index))
    return len(data), len(index), index


# +
def valley_detection(dataset, fs):
    window = []
    valleylist = []
    ybeat = []
    listpos = 0
    TH_elapsed = np.ceil(0.36 * fs)
    nvalleys = 0
    valleyarray = []
    
    localaverage = np.average(dataset)
    for datapoint in dataset:

        if (datapoint > localaverage) and (len(window) < 1):
            listpos += 1
        elif (datapoint <= localaverage):
            window.append(datapoint)
            listpos += 1
        else:
            minimum = min(window)
            beatposition = listpos - len(window) + (window.index(min(window)))
            valleylist.append(beatposition)
            window = []
            listpos += 1

    ## Ignore if the previous peak was within 360 ms interval becasuse it is T-wave
    for val in valleylist:
        if nvalleys > 0:
            prev_valley = valleylist[nvalleys - 1]
            elapsed = val - prev_valley
            if elapsed > TH_elapsed:
                valleyarray.append(val)
        else:
            valleyarray.append(val)
            
        nvalleys += 1    

    return valleyarray


def pair_valley(valley):
    pair_valley=[]
    for i in range(len(valley)-1):
        pair_valley.append([valley[i], valley[i+1]])
    return pair_valley


### peak detection ###

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.signal

# +
'''
Method 1 ) local minima and maxima
'''

def heo_threshold_peakdetection(dataset, fs):
    
    #print("dataset: ",dataset)
    window = []
    peaklist = []
    ybeat = []
    listpos = 0
    mean = np.average(dataset)
    TH_elapsed = np.ceil(0.36 * fs)
    npeaks = 0
    peakarray = []
    
    localaverage = np.average(dataset)
    for datapoint in dataset:

        if (datapoint < localaverage) and (len(window) < 1):
            listpos += 1
        elif (datapoint >= localaverage):
            window.append(datapoint)
            listpos += 1
        else:
            maximum = max(window)
            beatposition = listpos - len(window) + (window.index(max(window)))
            peaklist.append(beatposition)
            window = []
            listpos += 1

            
    ## Ignore if the previous peak was within 360 ms interval becasuse it is T-wave
    for val in peaklist:
        if npeaks > 0:
            prev_peak = peaklist[npeaks - 1]
            elapsed = val - prev_peak
            if elapsed > TH_elapsed:
                peakarray.append(val)
        else:
            peakarray.append(val)
            
        npeaks += 1    


    return peaklist

# +
'''
Method 2 ) first derivative with adaptive threshold

Reference:

1. Li, Bing Nan, Ming Chui Dong, and Mang I. Vai. "On an automatic delineator for 
arterial blood pressure waveforms." Biomedical Signal Processing and Control 5.1 (2010): 76-81.

2. Elgendi, Mohamed, et al. "Systolic peak detection in acceleration photoplethysmograms 
measured from emergency responders in tropical conditions." PLoS One 8.10 (2013).

'''

def seperate_division(data,fs):
    divisionSet = []
    for divisionUnit in range(0,len(data)-1,5*fs):  # index of groups (per 5sec) 
        eachDivision = data[divisionUnit: (divisionUnit+1) * 5 * fs]
        divisionSet.append(eachDivision)
    return divisionSet

def first_derivative_with_adaptive_ths(data, fs):
    
    peak = []
    divisionSet = seperate_division(data, fs)
    selectiveWindow = 2 * fs
    block_size = 5 * fs
    bef_idx = -300
    
    for divInd in range(len(divisionSet)):
        block = divisionSet[divInd]
        ths = np.mean(block[:selectiveWindow]) # ths: 2 seconds mean in block
        
        firstDeriv = block[1:] - block[:-1]
        for i in range(1,len(firstDeriv)):
            if  firstDeriv[i] <= 0 and firstDeriv[i-1] > 0:
                if block[i] > ths:
                    idx = block_size*divInd + i
                    if idx - bef_idx > (300*fs/1000):
                        peak.append(idx)
                        bef_idx = idx
                                                
    return peak
        



# +
'''
Method 3: Slope sum function with an adaptive threshold

Reference
1. Jang, Dae-Geun, et al. "A robust method for pulse peak determination 
in a digital volume pulse waveform with a wandering baseline." 
IEEE transactions on biomedical circuits and systems 8.5 (2014): 729-737.

2. Jang, Dae-Geun, et al. "A real-time pulse peak detection algorithm for 
the photoplethysmogram." International Journal of Electronics and Electrical Engineering 2.1 (2014): 45-49.
'''

def determine_peak_or_not(prevAmp, curAmp, nextAmp):
    if prevAmp < curAmp and curAmp >= nextAmp:
        return True
    else:
        return False
    
def onoff_set(peak, sig):     # move peak from dy signal to original signal   
    onoffset = []
    for p in peak:
        for i in range(p, 0,-1):
            if sig[i] == 0:
                onset = i
                break
        for j in range(p, len(sig)):
            if sig[j] == 0:
                offset = j
                break
        if onset < offset:
            onoffset.append([onset,offset])
    return onoffset
    

def slope_sum_function(data,fs):
    dy = [0]
    
    dy.extend(np.diff(data))
    #dy[dy < 0 ] = 0
    
    w = fs // 8
    dy_ = [0] * w
    for i in range(len(data)-w):
        sum_ = np.sum(dy[i:i+w])
        if sum_ > 0:
            dy_.append(sum_)
        else:
            dy_.append(0)
    
    init_ths = 0.6 * np.max(dy[:3*fs])
    ths = init_ths
    recent_5_peakAmp = []
    peak_ind = []
    bef_idx = -300
    
    for idx in range(1,len(dy_)-1):
        prevAmp = dy_[idx-1]
        curAmp = dy_[idx]
        nextAmp = dy_[idx+1]
        if determine_peak_or_not(prevAmp, curAmp, nextAmp) == True:
            if (idx - bef_idx) > (300 * fs /1000):  # Ignore if the previous peak was within 300 ms interval
                if len(recent_5_peakAmp) < 100:  
                    if curAmp > ths:
                        peak_ind.append(idx)
                        bef_idx = idx
                        recent_5_peakAmp.append(curAmp)
                elif len(recent_5_peakAmp) == 100:
                    ths = 0.7*np.median(recent_5_peakAmp)
                    if curAmp > ths:
                        peak_ind.append(idx)
                        bef_idx = idx
                        recent_5_peakAmp.pop(0)
                        recent_5_peakAmp.append(curAmp)
                        
    onoffset = onoff_set(peak_ind, dy_)
    corrected_peak_ind = []
    for onoff in onoffset:
        segment = data[onoff[0]:onoff[1]]
        corrected_peak_ind.append(np.argmax(segment) + onoff[0])
                    
    return corrected_peak_ind


# +
'''
Method 4

Event-Related Moving Averages with Dynamic Threshold

Reference

1. Elgendi, Mohamed, et al. "Systolic peak detection in acceleration photoplethysmograms 
measured from emergency responders in tropical conditions." PLoS One 8.10 (2013).

2. https://github.com/neuropsychology/NeuroKit/blob/8a2148fe477f20328d18b6da7bbb1c8438e60f18/neurokit2/signal/signal_formatpeaks.py

'''

def moving_average(signal, kernel='boxcar', size=5):
    size = int(size)
    window = scipy.signal.get_window(kernel, size)
    w = window / window.sum()
    
    # Extend signal edges to avoid boundary effects
    x = np.concatenate((signal[0] * np.ones(size), signal, signal[-1] * np.ones(size)))
    
    # Compute moving average
    smoothed = np.convolve(w, x, mode='same')
    smoothed = smoothed[size:-size]
    return smoothed


def moving_averages_with_dynamic_ths(signals,sampling_rate=64, peakwindow=.111, 
                                     beatwindow=.667, beatoffset=.02, mindelay=.3,show=False):
    if show:
        fig, (ax0, ax1) = plt.subplots(nrow=2, ncols=1, sharex=True)
        ax0.plot(data, label='filtered')
    
    signal = signals.copy()
    # ignore the samples with n
    signal[signal < 0] = 0
    sqrd = signal**2
    
    # Compute the thresholds for peak detection. Call with show=True in order
    # to visualize thresholds.
    ma_peak_kernel = int(np.rint(peakwindow * sampling_rate))
    ma_peak = moving_average(sqrd, size=ma_peak_kernel)
    
    ma_beat_kernel = int(np.rint(beatwindow * sampling_rate))
    ma_beat = moving_average(sqrd, size=ma_beat_kernel)

    
    thr1 = ma_beat + beatoffset * np.mean(sqrd)    # threshold 1

    if show:
        ax1.plot(sqrd, label="squared")
        ax1.plot(thr1, label="threshold")
        ax1.legend(loc="upper right")

    # Identify start and end of PPG waves.
    waves = ma_peak > thr1
    
    beg_waves = np.where(np.logical_and(np.logical_not(waves[0:-1]),
                                        waves[1:]))[0]
    end_waves = np.where(np.logical_and(waves[0:-1],
                                        np.logical_not(waves[1:])))[0]
    # Throw out wave-ends that precede first wave-start.
    end_waves = end_waves[end_waves > beg_waves[0]]

    # Identify systolic peaks within waves (ignore waves that are too short).
    num_waves = min(beg_waves.size, end_waves.size)
    min_len = int(np.rint(peakwindow * sampling_rate))    # threshold 2
    min_delay = int(np.rint(mindelay * sampling_rate))
    peaks = [0]

    for i in range(num_waves):

        beg = beg_waves[i]
        end = end_waves[i]
        len_wave = end - beg

        if len_wave < min_len: # threshold 2
            continue

        # Visualize wave span.
        if show:
            ax1.axvspan(beg, end, facecolor="m", alpha=0.5)

        # Find local maxima and their prominence within wave span.
        data = signal[beg:end]
        locmax, props = scipy.signal.find_peaks(data, prominence=(None, None))

        if locmax.size > 0:
            # Identify most prominent local maximum.
            peak = beg + locmax[np.argmax(props["prominences"])]
            # Enforce minimum delay between peaks(300ms)
            if peak - peaks[-1] > min_delay:
                peaks.append(peak)

    peaks.pop(0)

    if show:
        ax0.scatter(peaks, signal[peaks], c="r")

    peaks = np.asarray(peaks).astype(int)
    return peaks


# +
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def lmm_peakdetection(data,fs):
    
    peak_final = []
    peaks, _ = find_peaks(data,height=0)
    
    for peak in peaks:
        if data[peak] > 0:
            peak_final.append(peak)
        
    return peak_final



def ensemble_peak(preprocessed_data, fs, ensemble_ths=4):
    
    peak1 = heo_threshold_peakdetection(preprocessed_data,fs)
    peak2 = slope_sum_function(preprocessed_data, fs)
    peak3 = first_derivative_with_adaptive_ths(preprocessed_data, fs)
    peak4 = moving_averages_with_dynamic_ths(preprocessed_data)
    peak5 = lmm_peakdetection(preprocessed_data,fs)
    
    peak_dic = dict()

    for key in peak1:
        peak_dic[key] = 1

    for key in peak2:
        if key in peak_dic.keys():
            peak_dic[key] += 1
        else:
            peak_dic[key] = 1
    
    for key in peak3:
        if key in peak_dic.keys():
            peak_dic[key] += 1
        else:
            peak_dic[key] = 1
        
    for key in peak4:
        if key in peak_dic.keys():
            peak_dic[key] += 1
        else:
            peak_dic[key] = 1
        
    for key in peak5:
        if key in peak_dic.keys():
            peak_dic[key] += 1
        else:
            peak_dic[key] = 1
        
    peak_dic = dict(sorted(peak_dic.items()))

    count = 0
    cnt = 0
    bef_key = 0
    margin = 1

    new_peak_dic = dict()

    for key in peak_dic.keys():
        if cnt == 0:
            new_peak_dic[key] = peak_dic[key]
        else:
            if bef_key + margin >= key:  # 마진 1안에 다음 피크가 존재하면
                if peak_dic[bef_key] > peak_dic[key]: # 이전 피크 기준으로 개수 카운트
                    new_peak_dic[bef_key] += peak_dic[key]
                else:
                    #print("new peak dic: ",new_peak_dic)
                    new_peak_dic[key] = peak_dic[key] + peak_dic[bef_key] # 현재 피크 기준으로 개수 카운트
                    del(new_peak_dic[bef_key])
                    bef_key = key
            else:
                new_peak_dic[key] = peak_dic[key]
                bef_key = key
        cnt += 1
    
    ensemble_dic = dict()
    for (key, value) in new_peak_dic.items():
        if value >= ensemble_ths:
            ensemble_dic[key] = value
            
    final_peak = list(ensemble_dic.keys())
    
    return final_peak

### peak detection ###

# -*- coding: utf-8 -*-
# !pip install nolds

# +
import nolds

from scipy.interpolate import UnivariateSpline
from scipy import stats


# +
def heo_calc_RRI(peaklist, fs):
    RR_list = []
    RR_list_e = []
    cnt = 0
    while (cnt < (len(peaklist)-1)):
        RR_interval = (peaklist[cnt+1] - peaklist[cnt]) #Calculate distance between beats in # of samples
        ms_dist = ((RR_interval / fs) * 1000.0)  #fs로 나눠서 1초단위로 거리표현 -> 1ms단위로 change /  Convert sample distances to ms distances
        cnt += 1
        RR_list.append(ms_dist)
    mean_RR = np.mean(RR_list)

    for ind, rr in enumerate(RR_list):
        if rr >  mean_RR - 300 and rr < mean_RR + 300:
            RR_list_e.append(rr)
            
    RR_diff = []
    RR_sqdiff = []
    cnt = 0
    while (cnt < (len(RR_list_e)-1)):
        RR_diff.append(abs(RR_list_e[cnt] - RR_list_e[cnt+1]))
        RR_sqdiff.append(math.pow(RR_list_e[cnt] - RR_list_e[cnt+1], 2))
        cnt += 1
        
    return RR_list_e, RR_diff, RR_sqdiff

def heo_calc_heartrate(RR_list):
    HR = []
    heartrate_array=[]
    window_size = 10

    for val in RR_list:
        if val > 400 and val < 1500:
            heart_rate = 60000.0 / val #60000 ms (1 minute) / 한번 beat하는데 걸리는 시간
        # if RR-interval < .1905 seconds, heart-rate > highest recorded value, 315 BPM. Probably an error!
        elif (val > 0 and val < 400) or val > 1500:
            if len(HR) > 0:
                # ... and use the mean heart-rate from the data so far:
                heart_rate = np.mean(HR[-window_size:])

            else:
                heart_rate = 60.0
        else:
            # Get around divide by 0 error
            print("err")
            heart_rate = 0.0

        HR.append(heart_rate)

    return HR


# -

def heo_get_window_stats_original(ppg_seg, window_length, label=-1):  # Nan을 제외하고 평균 냄 
    
    fs = 64   
    
    peak = heo_threshold_peakdetection(ppg_seg, fs)
    RR_list, RR_diff, RR_sqdiff = heo_calc_RRI(peak, fs)
    
    # Time
    HR = heo_calc_heartrate(RR_list)
    HR_mean, HR_std = np.mean(HR), np.std(HR)
    SD_mean, SD_std = np.mean(RR_diff) , np.std(RR_diff)
    NN50 = [x for x in RR_diff if x > 50]
    pNN50 = len(NN50) / window_length
    bar_y, bar_x = np.histogram(RR_list)
    TINN = np.max(bar_x) - np.min(bar_x)
    RMSSD = np.sqrt(np.mean(RR_sqdiff))
    
    # Frequency
    rr_x = []
    pointer = 0
    for x in RR_list:
        pointer += x
        rr_x.append(pointer)
    RR_x_new = np.linspace(rr_x[0], rr_x[-1], int(rr_x[-1]))
    
    if len(rr_x) <= 5 or len(RR_list) <= 5:
        print("rr_x or RR_list less than 5")   
    
   
    interpolated_func = UnivariateSpline(rr_x, RR_list, k=3)
    
    datalen = len(RR_x_new)
    frq = np.fft.fftfreq(datalen, d=((1/1000.0)))
    frq = frq[range(int(datalen/2))]
    Y = np.fft.fft(interpolated_func(RR_x_new))/datalen
    Y = Y[range(int(datalen/2))]
    psd = np.power(Y, 2)  # power spectral density

    lf = np.trapz(abs(psd[(frq >= 0.04) & (frq <= 0.15)])) #Slice frequency spectrum where x is between 0.04 and 0.15Hz (LF), and use NumPy's trapezoidal integration function to find the are
    hf = np.trapz(abs(psd[(frq > 0.15) & (frq <= 0.5)])) #Do the same for 0.16-0.5Hz (HF)
    ulf = np.trapz(abs(psd[frq < 0.003]))
    vlf = np.trapz(abs(psd[(frq >= 0.003) & (frq < 0.04)]))
    
    if hf != 0:
        lfhf = lf/hf
    else:
        lfhf = 0
        
    total_power = lf + hf + vlf

    features = {'HR_mean': HR_mean, 'HR_std': HR_std, 'SD_mean': SD_mean, 'SD_std': SD_std, 'pNN50': pNN50, 'TINN': TINN, 'RMSSD': RMSSD,
                'LF': lf, 'HF': hf, 'ULF' : ulf, 'VLF': vlf, 'LFHF': lfhf, 'Total_power': total_power, 'label': label}

    return features


# +

def approximate_entropy(U, m=2, r=3):

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (N - m + 1.0)**(-1) * sum(np.log(C))

    N = len(U)

    return abs(_phi(m+1) - _phi(m))

def shannon_entropy(signal):
    #signal = list(signal)
    
    data_set = list(set(signal))
    freq_list = []
    for entry in data_set:
        counter = 0.
        for i in signal:
            if i == entry:
                counter += 1
        freq_list.append(float(counter) / len(signal))
        
    ent = 0.0
    for freq in freq_list:
        ent += freq * np.log2(freq)
    
    ent = -ent
    
    return ent



# https://horizon.kias.re.kr/12415/

def sample_entropy(sig,ordr,tor):
    # sig: the input signal or series, it should be numpy array with type float
    # ordr: order, the length of template,embedding dimension
    # tor: percent of standard deviation
    
    sig = np.array(sig)
    n = len(sig)
    #tor = np.std(sig)*tor
    
    matchnum = 0.0
    for i in range(n-ordr):
        tmpl = sig[i:i+ordr] # generate samples length ordr
        for j in range (i+1,n-ordr+1): 
            ltmp = sig[j:j+ordr]
            diff = tmpl-ltmp  # measure mean similarity
            if all(diff<tor):
                matchnum+=1
    
    allnum = (n-ordr+1)*(n-ordr)/2
    if matchnum<0.1:
        sen = 1000.0
    else:
        sen = -math.log(matchnum/allnum)
    return sen


# -

def heo_calc_td_hrv(RR_list, RR_diff, RR_sqdiff, window_length): 
    
    # Time
    HR = heo_calc_heartrate(RR_list)
    HR_mean, HR_std = np.mean(HR), np.std(HR)
    meanNN, SDNN, medianNN = np.mean(RR_list), np.std(RR_list), np.median(np.abs(RR_list))
    meanSD, SDSD = np.mean(RR_diff) , np.std(RR_diff)
    RMSSD = np.sqrt(np.mean(RR_sqdiff))
    
    NN20 = [x for x in RR_diff if x > 20]
    NN50 = [x for x in RR_diff if x > 50]
    pNN20 = len(NN20) / window_length
    pNN50 = len(NN50) / window_length
    
    
    bar_y, bar_x = np.histogram(RR_list)
    TINN = np.max(bar_x) - np.min(bar_x)
    
    RMSSD = np.sqrt(np.mean(RR_sqdiff))
    

    features = {'HR_mean': HR_mean, 'HR_std': HR_std, 'meanNN': meanNN, 'SDNN': SDNN, 'medianNN': medianNN,
                'meanSD': meanSD, 'SDSD': SDSD, 'RMSSD': RMSSD, 'pNN20': pNN20, 'pNN50': pNN50, 'TINN': TINN}

    return features


def calc_fd_hrv(RR_list):  
    
    rr_x = []
    pointer = 0
    for x in RR_list:
        pointer += x
        rr_x.append(pointer)
        
    if len(rr_x) <= 3 or len(RR_list) <= 3:
        print("rr_x or RR_list less than 5")   
        return 0
    
    RR_x_new = np.linspace(rr_x[0], rr_x[-1], int(rr_x[-1]))
    
   
    interpolated_func = UnivariateSpline(rr_x, RR_list, k=3)
    
    datalen = len(RR_x_new)
    frq = np.fft.fftfreq(datalen, d=((1/1000.0)))
    frq = frq[range(int(datalen/2))]
    Y = np.fft.fft(interpolated_func(RR_x_new))/datalen
    Y = Y[range(int(datalen/2))]
    psd = np.power(Y, 2)  # power spectral density

    lf = np.trapz(abs(psd[(frq >= 0.04) & (frq <= 0.15)])) #Slice frequency spectrum where x is between 0.04 and 0.15Hz (LF), and use NumPy's trapezoidal integration function to find the are
    hf = np.trapz(abs(psd[(frq > 0.15) & (frq <= 0.5)])) #Do the same for 0.16-0.5Hz (HF)
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
    bef_features = features
    
    return features


def calc_nonli_hrv(RR_list,label): 
    
    diff_RR = np.diff(RR_list)
    sd_heart_period = np.std(diff_RR, ddof=1) ** 2
    SD1 = np.sqrt(sd_heart_period * 0.5)
    SD2 = 2 * sd_heart_period - 0.5 * sd_heart_period
    pA = SD1*SD2
    
    if SD2 != 0:
        pQ = SD1 / SD2
    else:
        print("SD2 is zero")
        pQ = 0
    
    ApEn = approximate_entropy(RR_list,2,3)  
    shanEn = shannon_entropy(RR_list)
    #sampEn = nolds.sampen(RR_list,emb_dim=2)
    D2 = nolds.corr_dim(RR_list, emb_dim=2)
    #dfa1 = nolds.dfa(RR_list, range(4,17))
    # dfa2 = nolds.dfa(RR_list, range(16,min(len(RR_list)-1, 66)))
    #dimension, delay, threshold, norm, minimum_diagonal_line_length = 3, 2, 0.7, "manhattan", 2
    #rec_mat = recurrence_matrix(RR_list, dimension, delay, threshold, norm)
    #REC, RPImean, RPImax, RPadet = recurrence_quantification_analysis(rec_mat, minimum_diagonal_line_length)
    # recurrence_rate, average_diagonal_line_length, longest_diagonal_line_length, determinism

    features = {'SD1': SD1, 'SD2': SD2, 'pA': pA, 'pQ': pQ, 'ApEn' : ApEn, 'shanEn': shanEn, 'D2': D2, 
                'label': label}
    # 'dfa1': dfa1, 'dfa2': dfa2, 'REC': REC, 'RPImean': RPImean, 'RPImax': RPImax, 'RPadet': RPadet,
    return features


def heo_get_window_stats_27_features(ppg_seg, window_length, label, ensemble, ma_usage):  
    
    fs = 64  
    
    if ma_usage:
        fwd = moving_average(ppg_seg, size=3)
        bwd = moving_average(ppg_seg[::-1], size=3)
        ppg_seg = np.mean(np.vstack((fwd,bwd[::-1])), axis=0)
    ppg_seg = np.array([item.real for item in ppg_seg])
    
    #peak = threshold_peakdetection(ppg_seg, fs)
    #peak = first_derivative_with_adaptive_ths(ppg_seg, fs)
    #peak = slope_sum_function(ppg_seg, fs)
    #peak = moving_averages_with_dynamic_ths(ppg_seg)
    peak = lmm_peakdetection(ppg_seg,fs)

        
    if ensemble:
        ensemble_ths = 3
        #print("one algorithm peak length: ", len(peak))
        peak = ensemble_peak(ppg_seg, fs, ensemble_ths)
        #print("after ensemble peak length: ", len(peak))
        
        if(len(peak) < 100):
            print("skip")
            return []

        
    RR_list, RR_diff, RR_sqdiff = heo_calc_RRI(peak, fs)
    #print(RR_list)
    
    if len(RR_list) <= 3:
        return []
    
    td_features = heo_calc_td_hrv(RR_list, RR_diff, RR_sqdiff, window_length)
    fd_features = calc_fd_hrv(RR_list)
    
    if fd_features == 0:
        return []
    nonli_features = calc_nonli_hrv(RR_list,label)
    
    total_features = {**td_features, **fd_features, **nonli_features}
    
    
    return total_features

