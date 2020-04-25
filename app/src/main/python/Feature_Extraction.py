
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
import os

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
    j = 0
    for i in peaklist:
        readable[j] = datetime.datetime.fromtimestamp(timestamp[i]/1000)
        j+=1

    return readable

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

def calc_td_hrv(RR_list, RR_diff, RR_sqdiff):
    sdnn = np.std(RR_list)

    sdsd = np.std(RR_diff)
    NN20 = [x for x in RR_diff if x > 20]
    NN50 = [x for x in RR_diff if x > 50]

    rmssd=np.sqrt(np.mean(RR_sqdiff))

    pnn20 = (len(NN20))/len(RR_diff)
    pnn50 = (len(NN50))/len(RR_diff)

    return sdnn, sdsd, rmssd, pnn20, pnn50


def calc_fd_hrv(RR_list):
    rr_x = []
    pointer = 0
    for x in RR_list:
        pointer += x
        rr_x.append(pointer)
    RR_x_new = np.linspace(rr_x[0], rr_x[-1], int(rr_x[-1]))

#     f = interp1d(RR_x, RR_y, kind='cubic') #Interpolate the signal with cubic spline interpolation
    interpolated_func = UnivariateSpline(rr_x, RR_list, k=3)

    datalen = len(RR_x_new)
    frq = np.fft.fftfreq(datalen, d=((1/1000.0)))
    frq = frq[range(int(datalen/2))]
    Y = np.fft.fft(interpolated_func(RR_x_new))/datalen
    Y = Y[range(int(datalen/2))]
    psd = np.power(Y, 2)

    lf = np.trapz(abs(psd[(frq >= 0.04) & (frq <= 0.15)])) #Slice frequency spectrum where x is between 0.04 and 0.15Hz (LF), and use NumPy's trapezoidal integration function to find the are
    hf = np.trapz(abs(psd[(frq >= 0.16) & (frq <= 0.5)])) #Do the same for 0.16-0.5Hz (HF)
    lfhf = lf/hf

    return lf, hf, lfhf

def cal_nonli_hrv(RR_list):

    diff_RR = np.diff(RR_list)

    sd1 = np.sqrt(np.std(diff_RR, ddof=1) ** 2 * 0.5)
    sd2 = np.sqrt(2 * np.std(RR_list, ddof=1) ** 2 - 0.5 * np.std(diff_RR, ddof=1) ** 2)
    sd1sd2 = sd1/sd2

    return sd1, sd2, sd1sd2




def PPG_feature(Raw_PPG, Timestamp_PPG, fs, smoothing_period=80):
    ppg_lowcut = 0.5 #fmin
    ppg_highcut = 1.5 #fmax

    ## Filtering ##
    ppg_filter = butter_bandpassfilter(Raw_PPG, ppg_lowcut, ppg_highcut, fs, order=2)
    ppg_smooth = movingaverage(ppg_filter, periods=smoothing_period)
    ## Peak detection ##
    ppg_peaklist_t = threshold_peakdetection(ppg_smooth, fs)

    ## Correct peaklist## ppg_correct_peaklist_t is the list of indexes for ppg peak points
    ppg_correct_peaklist_t = correct_peaklist(Raw_PPG, ppg_peaklist_t, fs)

    ## RR intervals ## They are in: ms, ms and ms^2 respectively
    ppgT_RR_list, ppgT_RR_diff, ppgT_RR_sqdiff = calc_RRI(ppg_correct_peaklist_t, fs)

    ## GET TIME DATA ##
    ppg_time = getTime(Timestamp_PPG, ppg_correct_peaklist_t)

    ## Heart Rate ##
    ppgT_HR = calc_heartrate(ppgT_RR_list)
    rej_ppgT_HR = reject_outliers(ppgT_HR,  deviation_threshold=3)
#    ppg_bpm = calc_bpm(rej_ppgT_HR)

    ## HRV - TIME ##
    ppg_sdnn, ppg_sdsd, ppg_rmssd, ppg_pnn20, ppg_pnn50 = calc_td_hrv(ppgT_RR_list, ppgT_RR_diff, ppgT_RR_sqdiff)
    ## HRV - FREQ ##
    ppg_lf, ppg_hf, ppg_lfhf = calc_fd_hrv(ppgT_RR_list)
    ## HRV - NONLINEAR ##
    ppg_sd1, ppg_sd2, ppg_sd1sd2 = cal_nonli_hrv(ppgT_RR_list)


    PPG_TIME = str(ppg_time[0])
    PPG_HR = np.average(rej_ppgT_HR)
#    PPG_BPM = np.average(ppg_bpm)
    PPG_SDNN = np.average(ppg_sdnn)
    PPG_SDSD = np.average(ppg_sdsd)
    PPG_RMSSD = np.average(ppg_rmssd)
    PPG_PNN20 = np.average(ppg_pnn20)
    PPG_PNN50 = np.average(ppg_pnn50)
    PPG_LF = np.average(ppg_lf)/(10**6)         #scaling constant
    PPG_HF = np.average(ppg_hf)/(10**5)         #scaling constant
    PPG_LFHF = np.average(ppg_lfhf)
    PPG_SD1 = np.average(ppg_sd1)/(10**3)
    PPG_SD2 = np.average(ppg_sd2)/(10**3)
    PPG_SD1SD2 = np.average(ppg_sd1sd2)


    return PPG_TIME, PPG_HR, PPG_SDNN, PPG_SDSD, PPG_RMSSD, PPG_PNN20, PPG_PNN50, PPG_LF, PPG_HF, PPG_LFHF, PPG_SD1, PPG_SD2, PPG_SD1SD2

def dens_val(index, density):
    d = density.copy()
    ind = index.copy()
    for i in range(len(index)):
        d = d[ind.pop(0)]
    return d
#%%

def Sample_Locator(Sample, bndrs):
    """
    Sample: Comes from the Feature_Extraction Function
    bndrs: Are two constant M*N and M*(N+1) np arrays which are directly read from the phone memory. Set as constants here for now.
    """
    m = len(Sample) #number of features

    index = []
    for i in range(m):
        for j in range(len(bndrs[i])):
            if Features[i]<bndrs[i,j] :
                break
        if Features[i]>=bndrs[i,j]:
            j+=1
        index.append(j)

    return index #returns the index of Sample on the grid



# In[6]:


def main(DAY=0.5):
    url1 = 'data_201909121857.csv'

    Raw_PPG = openShimmerFile(url1, 'ppg')
    Timestamp_PPG = openShimmerFile(url1, 'timestamp')

    Features = PPG_feature(Raw_PPG, Timestamp_PPG, fs=20, smoothing_period=10)
    Sample = [Features[2], Features[7], Features[9], Features[10]]

    with open(os.environ["HOME"]+'Samples.csv', 'a', newline='') as file:
        file_writer = csv.writer(file, delimiter=',')
        file_writer.writerow(Sample)

    t = False    #TRIGGER SIGNAL, OUTPUT, set to False as default

    if DAY<0:
        raise ValueError("DAY should be a non-negative float number")

    elif DAY==1:
        stored_data = np.genfromtxt(os.environ["HOME"]+'Samples.csv',delimiter=',')
        Mean = stored_data.mean(axis=0)
        STD = stored_data.std(axis=0)
        bndrs = np.array((Mean-STD/2, Mean+STD/2)).T
        density = np.zeros(([bndrs.shape[1]+1]*(bndrs.shape[0])))

        for row in stored_data:
            index = Sample_Locator(Sample, bndrs)
            density[index]+=1
    #Save Density and bndrs

    elif DAY>1:
    #load density and bndrs
        index= Sample_Locator(Features, bndrs)
        d = dens_val(a, density)
        d_cal = d/density.max()
        d_cal= max(d_cal, 0.05)
        eps = np.random.random()

        if eps<d_cal:
            t = True

    return t