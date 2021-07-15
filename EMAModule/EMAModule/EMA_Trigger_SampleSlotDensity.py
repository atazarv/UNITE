import csv
import numpy as np
import os
from pathlib import Path
import pandas as  pd
import heartpy as hp
from datetime import datetime
import json


class EMATriggerDensity():
    def __init__(self, raw_data, ppg_data, user_id, filepath):
        self.user_id = user_id
        self.filepath = filepath # to get previous samples to process
        self.raw_data = raw_data
        self.ppg_data = ppg_data

    def Sample_Locator(self, Sample, bndrs):
        m = len(Sample)
        index = []
        for i in range(m):
            for j in range(len(bndrs[i])):
                if Sample[i]<bndrs[i,j] :
                    break
            if Sample[i]>=bndrs[i,j]:
                j+=1
            index.append(j)

        return index

    def return_trigger(self):
        t = False

        Sample_file_path = self.filepath / ('Sample_'+self.user_id+'.csv')

        try:
            sample_count = 0
            with open(Sample_file_path) as file:
                for _ in file:
                    sample_count += 1
        except:
            sample_count = 0

        if not(sample_count%100) and sample_count:
            stored_data = np.genfromtxt(Sample_file_path,delimiter=',')[1:,1:-4]

            Mean = stored_data.mean(axis=0)
            STD = stored_data.std(axis=0)
            bndrs = np.array((Mean-STD/2, Mean+STD/2)).T
            density = np.zeros(([bndrs.shape[1]+1]*(bndrs.shape[0])))

            for row in stored_data:
                index = self.Sample_Locator(row, bndrs)
                density[tuple(index)]+=1
            np.save(self.filepath / ('density_'+self.user_id), density)
            np.savetxt(self.filepath / ('bndrs_'+self.user_id+'.csv'), bndrs, delimiter=',')


        elif sample_count>100: #and not(sleep)
            density = np.load(self.filepath / ('density_'+self.user_id+'.npy'))
            bndrs = np.genfromtxt(self.filepath / ('bndrs_'+self.user_id+'.csv'), delimiter=',')
            index= self.Sample_Locator(self.ppg_data, bndrs)
            d_cal = density[tuple(index)]/density.max()
            d_cal= max(d_cal, 0.15)

            eps = np.random.random()
            if eps<d_cal:
                t = True

        return t

#%%
if __name__ == "__main__":

    datapath = '/Users/james/Documents/UCI_Research/MicroRCT/Github/UNITE/EMA_Triggering_Module/test_data/data_uniterct552-2020-11-09-08-41-01.csv'
    filepath = Path('/Users/james/Documents/UCI_Research/MicroRCT/Github/UNITE/EMA_Triggering_Module/test_data/test_file_path')
    user_id = 'uniterct552'
    data = pd.read_csv(datapath, header=0, delimiter='\t')

