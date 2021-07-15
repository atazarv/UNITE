import csv
import numpy as np
import os
from pathlib import Path
import pandas as  pd
import heartpy as hp
from datetime import datetime
import json


class EMATriggerRestTime():
    def __init__(self, raw_data, ppg_data, user_id, filepath):
        self.user_id = user_id
        self.filepath = filepath # to get previous samples to process
        self.raw_data = raw_data
        self.ppg_data = ppg_data


    def return_trigger(self):
        rest_time = datetime.fromtimestamp(self.raw_data.timestamp.iloc[-1]/1000).hour < 7

        t = False

        if not rest_time:
            t = True

        return t

#%%
if __name__ == "__main__":
    datapath = '/Users/james/Documents/UCI_Research/MicroRCT/Github/UNITE/EMA_Triggering_Module/test_data/data_uniterct552-2020-11-09-08-41-01.csv'
    filepath = Path('/Users/james/Documents/UCI_Research/MicroRCT/Github/UNITE/EMA_Triggering_Module/test_data/test_file_path')
    user_id = 'uniterct552'
    data = pd.read_csv(datapath, header=0, delimiter='\t')

