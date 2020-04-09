
# coding: utf-8

import csv
import datetime
#import matplotlib.pyplot as plt
#import matplotlib.patches as mpatches
from scipy.signal import butter, lfilter
import numpy as np
import math
from scipy.interpolate import UnivariateSpline
from os.path import dirname, join

#%%

def openShimmerFile(url, column_name):
    req_data = []

    # Read File
    with open(join(dirname(__file__), url)) as f:
        reader = csv.reader(f, delimiter = '\t')
        # Store data in lists
        sep = reader.__next__()
        data_header = reader.__next__()

        index = -1
        for i in range(len(data_header)):
            if (data_header[i] == column_name):
                index = i

        if (index < 0):
            raise ColumnNotFound
        
        for row in reader:
            if (index <= len(row)):
                req_data.append(float(row[index]))

    return req_data

#%%

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpassfilter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def movingaverage(data, periods=4):
    result = []
    data_set = np.asarray(data)
    weights = np.ones(periods) / periods
    result = np.convolve(data_set, weights, mode='valid')
    return result

def threshold_peakdetection(dataset, fs):
    window = []
    peaklist = []
    ybeat = []
    listpos = 0
    TH_elapsed = np.ceil(0.36 * fs)
    npeaks = 0
    s = 0
    peakarray = []
    
    localaverage = np.average(dataset)
    for datapoint in dataset:

        if (datapoint < localaverage) and (len(window) < 1):
            listpos += 1
        elif (datapoint > localaverage):
            window.append(datapoint)
            listpos += 1
        else:
            maximum = max(window)
            beatposition = listpos - len(window) + (window.index(maximum))
            peaklist.append(beatposition)
            window = []
            listpos += 1
      
    ## Ignore if the previous peak was within 360 ms interval becasuse it is T-wave
    for val in peaklist:
        if npeaks > 0:
            prev_peak = peaklist[npeaks - 1 - s]
            elapsed = val - prev_peak
#            if npeaks>43:
#                print(npeaks, elapsed)
            s = int((elapsed<=TH_elapsed) or (dataset[peaklist[npeaks]]<0.2*dataset[peaklist[npeaks-1]]))
            if not(s):
                peakarray.append(val)
        npeaks += 1    
    
    ybeat = [dataset[x] for x in peakarray]

    return peakarray#, peaklist

def correct_peaklist(signal, peaklist, fs, tol=0.05):
    tol = int(tol * fs)
    length = len(signal)

    newR = []
    for r in peaklist:
        a = r - tol
        if a < 0:
            continue
        b = r + tol
        if b > length:
            break
        newR.append(a + np.argmax(signal[a:b]))

    return newR

def calc_RRI(peaklist, fs):
    RR_list = []
    RR_list_merge = []
    cnt = 0
    while (cnt < (len(peaklist)-1)):
        RR_interval = (peaklist[cnt+1] - peaklist[cnt])
        ms_dist = ((RR_interval / fs) * 1000.0) 
        RR_list.append(ms_dist)
        cnt += 1

    RR_diff = []
    RR_sqdiff = []
    cnt = 0
    while (cnt < (len(RR_list)-1)):
        RR_diff.append(abs(RR_list[cnt] - RR_list[cnt+1]))
        RR_sqdiff.append(math.pow(RR_list[cnt] - RR_list[cnt+1], 2))
        cnt += 1
        
    return RR_list, RR_diff, RR_sqdiff

def getTime(timestamp, peaklist):
    readable = [0 for x in range(int(len(peaklist))) ]
    timearray = []
    j = 0 
    for i in peaklist:
        readable[j] = datetime.datetime.fromtimestamp(timestamp[i]/1000).strftime('%M')
        j+=1
    
    for i in range(len(readable)):
        timearray.append(int(readable[i]))
        
    return timearray

def calc_heartrate(RR_list):
    HR = []
    heartrate_array=[]
    window_size = 10

    for val in RR_list:
        if val > 400 and val < 1500:
            heart_rate = 60000.0 / val
        # if RR-interval < .1905 seconds, heart-rate > highest recorded value, 315 BPM. Probably an error!
        elif (val > 0 and val < 400) or val > 1500:
            if len(HR) > 0:
                # ... and use the mean heart-rate from the data so far:
                heart_rate = np.mean(HR[-window_size:])

            else:
                heart_rate = 60.0
        else:
            # Get around divide by 0 error
            heart_rate = 0.0

        HR.append(heart_rate)

    return HR

def reject_outliers(data, deviation_threshold=0.67, fill_with='average'):
    result = []

    mean_data = np.mean(data)

    if (fill_with == 'average'):
        filler = np.mean(data)
    elif (fill_with == 'zero'):
        filler = 0.0
    elif (fill_with == 'None'):
        filler = None

    stdData = np.std(data)
    for i in range(len(data)):
        if (abs(data[i] - mean_data) < (deviation_threshold * stdData)):
            result.append(data[i])
        else:
            result.append(filler)
    if len(result) == 0:
        result = data
    return result

def calc_td_hrv(RR_list, RR_diff, RR_sqdiff, f_s):
#     ibi=[0 for x in range(int(len(RR_list))+1)]
    ibi=[0 for x in range(int(len(RR_list)))]
    sndd=[0 for x in range(int(len(RR_list)))]
    sdsd=[0 for x in range(int(len(RR_diff)))]
    rmssd=[0 for x in range(int(len(RR_sqdiff)))]
    NN20=[0 for x in range(int(len(RR_diff)))]
    NN50=[0 for x in range(int(len(RR_diff)))]


    window_length = 300
    
    dt = np.dtype('Float64')
    ibi_array = np.array(ibi, dtype=dt)
    sdnn_array = np.array(sndd, dtype=dt)
    sdsd_array = np.array(sdsd, dtype=dt)
    rmssd_array = np.array(rmssd, dtype=dt)

    for i in range(len(RR_list)):
        if(len(RR_list[i:i+window_length])==window_length):
            ibi_array[i]=np.mean(RR_list[i:i+window_length])
            sdnn_array[i]=(np.std(RR_list[i:i+window_length]))
        else:
            ibi_array[i]=np.mean(RR_list[i:])
            sdnn_array[i]=(np.std(RR_list[i:]))
            
    for i in range(len(RR_diff)):
        if(len(RR_diff[i:i+window_length])==window_length):
            sdsd_array[i]=np.std(RR_diff[i:i+window_length])
            NN20[i] = [x for x in RR_diff[i:i+window_length] if x > 20]
            NN50[i] = [x for x in RR_diff[i:i+window_length] if x > 50]
        else:
            sdsd_array[i]=np.std(RR_diff[i:])
            NN20[i] = [x for x in RR_diff[i:] if x > 20]
            NN50[i] = [x for x in RR_diff[i:] if x > 50]

    for i in range(len(RR_sqdiff)):
        if(len(RR_sqdiff[i:i+window_length])==window_length):
            rmssd_array[i]=np.sqrt(np.mean(RR_sqdiff[i:i+window_length]))
        else:
            rmssd_array[i]=np.sqrt(np.mean(RR_sqdiff[i:]))

        
    pnn20=[0 for x in range(int(len(NN20)))]
    pnn50=[0 for x in range(int(len(NN50)))]
    
    
    for i in range(len(NN20)):
        pnn20[i] = (len(NN20[i]))/window_length
    
    for i in range(len(NN50)):
        pnn50[i]=(len(NN50[i]))/window_length
    
    
    return sdnn_array, sdsd_array, rmssd_array, pnn20, pnn50


def calc_fd_hrv(RR_list, fs):  
    window = 300 # minimun range of frequency domain HRV is 5 minutes
    lf = [0 for x in range(int(len(RR_list))+1)]
    hf = [0 for x in range(int(len(RR_list))+1)]
    lfhf = [0 for x in range(int(len(RR_list))+1)]
    
    for i in range(0, len(RR_list), window):
        rr_x = []
        pointer = 0
        for x in RR_list[i:i+window]:
            pointer += x
            rr_x.append(pointer)
        RR_x_new = np.linspace(rr_x[0], rr_x[-1], rr_x[-1])
    
    
#     f = interp1d(RR_x, RR_y, kind='cubic') #Interpolate the signal with cubic spline interpolation
        interpolated_func = UnivariateSpline(rr_x, RR_list[i:i+window], k=3)

        datalen = len(RR_x_new)
        frq = np.fft.fftfreq(datalen, d=((1/1000.0)))
        frq = frq[range(int(datalen/2))]
        Y = np.fft.fft(interpolated_func(RR_x_new))/datalen
        Y = Y[range(int(datalen/2))]
        psd = np.power(Y, 2)

        lf[i] = np.trapz(abs(psd[(frq >= 0.04) & (frq <= 0.15)])) #Slice frequency spectrum where x is between 0.04 and 0.15Hz (LF), and use NumPy's trapezoidal integration function to find the are
        hf[i] = np.trapz(abs(psd[(frq >= 0.16) & (frq <= 0.5)])) #Do the same for 0.16-0.5Hz (HF)
        lfhf[i] = lf[i]/hf[i]
        
    for i in range(0, len(lf)):
        if (lf[i]==0):
            lf[i] = lf[i-1]

    for i in range(0, len(hf)):
        if (hf[i]==0):
            hf[i] = hf[i-1]

    for i in range(0, len(lfhf)):
        if (lfhf[i]==0):
            lfhf[i] = lfhf[i-1]

    return lf, hf, lfhf

def cal_nonli_hrv(RR_list, fs):

    diff_RR = np.diff(RR_list)

    sd1 = np.sqrt(np.std(diff_RR, ddof=1) ** 2 * 0.5)
    sd2 = np.sqrt(2 * np.std(RR_list, ddof=1) ** 2 - 0.5 * np.std(diff_RR, ddof=1) ** 2)
    sd1sd2 = sd1/sd2

    return sd1, sd2, sd1sd2

def truncate(n, decimals=2):
    multiplier = 10**decimals
    return int(n*multiplier)/multiplier


def PPG_feature(Raw_PPG, Timestamp_PPG, fs):
    ppg_lowcut = 0.5 #fmin
    ppg_highcut = 1.5 #fmax
    
    ## Filtering ## 
    ppg_filter = butter_bandpassfilter(Raw_PPG, ppg_lowcut, ppg_highcut, fs, order=2)
    ppg_smooth = movingaverage(ppg_filter, periods=80)
    ## Peak detection ##
    ppg_peaklist_t = threshold_peakdetection(ppg_smooth, fs)

    ## Correct peaklist##
    ppg_correct_peaklist_t = correct_peaklist(Raw_PPG, ppg_peaklist_t, fs)
    ## RR intervals ##
    ppgT_RR_list, ppgT_RR_diff, ppgT_RR_sqdiff = calc_RRI(ppg_correct_peaklist_t, fs)

    ## GET TIME DATA ##
    ppg_time = getTime(Timestamp_PPG, ppg_correct_peaklist_t)
    
    ## Heart Rate ##
    ppgT_HR = calc_heartrate(ppgT_RR_list)
    rej_ppgT_HR = reject_outliers(ppgT_HR,  deviation_threshold=3)
#    ppg_bpm = calc_bpm(rej_ppgT_HR)
    
    ## HRV - TIME ##
    ppg_sdnn, ppg_sdsd, ppg_rmssd, ppg_pnn20, ppg_pnn50 = calc_td_hrv(ppgT_RR_list, ppgT_RR_diff, ppgT_RR_sqdiff, fs)
    ## HRV - FREQ ##
    ppg_lf, ppg_hf, ppg_lfhf = calc_fd_hrv(ppgT_RR_list, fs)
    ## HRV - NONLINEAR ##
    ppg_sd1, ppg_sd2, ppg_sd1sd2 = cal_nonli_hrv(ppgT_RR_list, fs)
    

    PPG_TIME = ppg_time[0]
    PPG_HR = np.average(rej_ppgT_HR)
#    PPG_BPM = np.average(ppg_bpm)
    PPG_SDNN = np.average(ppg_sdnn)
    PPG_SDSD = np.average(ppg_sdsd)
    PPG_RMSSD = np.average(ppg_rmssd)
    PPG_PNN20 = np.average(ppg_pnn20)
    PPG_PNN50 = np.average(ppg_pnn50)
    PPG_LF = np.average(ppg_lf)
    PPG_HF = np.average(ppg_hf)
    PPG_LFHF = np.average(ppg_lfhf)
    PPG_SD1 = np.average(ppg_sd1)
    PPG_SD2 = np.average(ppg_sd2)
    PPG_SD1SD2 = np.average(ppg_sd1sd2)


    PPG_TIME= truncate(PPG_TIME)
    PPG_HR= truncate(PPG_HR)
    PPG_SDNN= truncate(PPG_SDNN)
    PPG_SDSD= truncate(PPG_SDSD)
    PPG_RMSSD= truncate(PPG_RMSSD)
    PPG_PNN20= truncate(PPG_PNN20)
    PPG_PNN50= truncate(PPG_PNN50)
    PPG_LF= truncate(PPG_LF)
    PPG_HF= truncate(PPG_HF)
    PPG_LFHF= truncate(PPG_LFHF)
    PPG_SD1= truncate(PPG_SD1)
    PPG_SD2= truncate(PPG_SD2)
    PPG_SD1SD2= truncate(PPG_SD1SD2)

                  
    return PPG_TIME, PPG_HR, PPG_SDNN, PPG_SDSD, PPG_RMSSD, PPG_PNN20, PPG_PNN50, PPG_LF, PPG_HF, PPG_LFHF, PPG_SD1, PPG_SD2, PPG_SD1SD2
                  


# In[6]:


def Feature_Extraction():
    url1  = "data_201909121857.csv"
    Raw_PPG = openShimmerFile(url1, 'ppg')
    Timestamp_PPG = openShimmerFile(url1, 'timestamp')
    return PPG_feature(Raw_PPG, Timestamp_PPG, 20)