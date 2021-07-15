import pandas as  pd
import numpy as np


class EMATriggerWatch():
    def __init__(self, raw_data, ppg_data, user_id, filepath):
        self.data = raw_data

    def wear_detect(self, x=1,y=1,z=9.65, stdev=0.35):
        data = self.data
        acc_std = np.linalg.norm(data[['accx','accy','accz']], axis=1)[:-2].std()
        accx = data['accx'].values[:-1].mean()
        accy = data['accy'].values[:-1].mean()
        accz = data['accz'].values[:-1].mean()

        not_worn = (acc_std < stdev) & (np.abs(accz) > z) & (np.abs(accx) < x) & (np.abs(accy) <y)

        return not_worn

    def return_trigger(self):
        return self.wear_detect()
