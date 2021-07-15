import csv
import numpy as np
import os
from pathlib import Path
import pandas as  pd
import heartpy as hp
from datetime import datetime
import json


class EMATriggerLastTriggerTime():
    def __init__(self, raw_data, ppg_data, user_id, filepath):
        self.user_id = user_id
        self.filepath = filepath # to get previous samples to process
        self.raw_data = raw_data
        self.ppg_data = ppg_data
        self.json_filename = self.filepath / ('userinfo_'+self.user_id+'.json')
        try:
            self.user_info = json.load(open(self.json_filename))
        except: # in case jsonfile goes bad
            self.user_info = {}


    def return_trigger(self):
        current_date = datetime.fromtimestamp(self.raw_data.timestamp.iloc[-1]/1000)
        current_timestamp = current_date.timestamp()
        t = False

        if 'lastTrigger' not in self.user_info:
            t = True
            self.user_info['lastTrigger'] = current_timestamp
        else:
            # 2 hours as the minimum between EMA triggers
            if (current_timestamp - self.user_info['lastTrigger']) >= 7200:
                t = True
                self.user_info['lastTrigger'] = current_timestamp

        if current_date.hour < 7:
            t = False

        with open(self.json_filename, 'w') as f:
            f.write(json.dumps(self.user_info))

        return t

#%%
if __name__ == "__main__":
    datapath = '/Users/james/Documents/UCI_Research/MicroRCT/Github/UNITE/EMA_Triggering_Module/test_data/data_uniterct552-2020-11-09-08-41-01.csv'
    filepath = Path('/Users/james/Documents/UCI_Research/MicroRCT/Github/UNITE/EMA_Triggering_Module/test_data/test_file_path')
    user_id = 'uniterct552'
    data = pd.read_csv(datapath, header=0, delimiter='\t')

