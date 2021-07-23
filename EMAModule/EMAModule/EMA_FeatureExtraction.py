import csv
import numpy as np
import os
from pathlib import Path
import pandas as  pd
import heartpy as hp
from datetime import datetime

class FeatureExtractionModule():
    def __init__(self, raw_data):
        self.data = raw_data

    def PPG_features(self, data):
        raw = data.ppg.values
        timer = data.timestamp.values
        sample_rate = hp.get_samplerate_mstimer(timer)
        filtered = hp.filter_signal(raw, [0.7, 3.5], sample_rate=sample_rate, order=3, filtertype='bandpass')
        wd, m = hp.process(filtered, sample_rate = sample_rate, clean_rr=True, calc_freq=False)

        return m

    def FeatureExtract(self):
        ppg_features_dict = self.PPG_features(self.data[['ppg', 'timestamp']])
        ppg_features = list(ppg_features_dict.values())
        if (ppg_features.count(np.nan) or (ppg_features.count(0)>2) or ((np.sum(ppg_features)/(10e10))>1)):
            print('Error extracting feature.')
            print('NaN count', ppg_features.count(np.nan))
            print('0 count', ppg_features.count(0))
            print('Sum feature / 10e10', np.sum(ppg_features)/(10e10))
            print('Feature list length', len(ppg_features))
            print('Feature list', (ppg_features))

            raise ValueError("Returned error while extracting features")

        return list(ppg_features_dict.keys()), ppg_features

if __name__ == '__main__':

    datapath = '/Users/james/Documents/UCI_Research/MicroRCT/Github/UNITE/EMA_Triggering_Module/test_data/data_uniterct552-2020-11-09-08-41-01.csv'
    data = pd.read_csv(datapath, header=0, delimiter='\t')

    mod = FeatureExtractionModule(data)
    mod.FeatureExtract()
