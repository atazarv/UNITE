import csv
import numpy as np
import os
from pathlib import Path
import pandas as  pd
import heartpy as hp
from datetime import datetime


#%% For whole signal (2minutes) analysis only
    
def PPG_features(data):
    raw = data.ppg.values
    timer = data.timestamp.values
    sample_rate = hp.get_samplerate_mstimer(timer)
    filtered = hp.filter_signal(raw, [0.7, 3.5], sample_rate=sample_rate, order=3, filtertype='bandpass')
    wd, m = hp.process(filtered, sample_rate = sample_rate, clean_rr=True, calc_freq=False)
    
    return m

def Sample_Locator(Sample, bndrs):
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


def wear_detect(data, x=1,y=1,z=9.65, stdev=0.35):
    acc_std = np.linalg.norm(data[['accx','accy','accz']], axis=1)[:-2].std()
    accx = data['accx'].values[:-1].mean()
    accy = data['accy'].values[:-1].mean()
    accz = data['accz'].values[:-1].mean()
    
    not_worn = (acc_std < stdev) & (np.abs(accz) > z) & (np.abs(accx) < x) & (np.abs(accy) <y)
    
    return not_worn

#%%

def main(datapath, filespath, user_id, realtime=True, sleep = False): 
	#filespath:            directory in which distribution files are or will be stored
    #datafile:        Address to the new coming 2m window signals including ppg
	#user_id:         e.g. "uniterct446"
    #realtime:        whether the signal is real time or it's delayed signal, which was stored on the watch
    #sleep:           Whether the user is asleep or not
    
    #do not notify between midnight and 7am:
    #rest_time = datetime.strptime(str(datapath)[-23:-4], '%Y-%m-%d-%H-%M-%S').hour < 7
    #load data
    data = pd.read_csv(datapath, header=0, delimiter='\t')
    
    
    watch_not_worn = wear_detect(data)
    if watch_not_worn:
        #print("*************************Watch Not Worn", datapath)
        raise ValueError("The watch is not worn")
    
    data = data[['ppg', 'timestamp']]
    rest_time = datetime.fromtimestamp(data.timestamp.iloc[-1]/1000).hour < 7
    #exclude too short or too long:
    if data.shape[0]<5800:
        raise ValueError("Sample is too short")
    if data.shape[0]>6200:
#        print(data.shape)
        raise ValueError("Sample is too long")
    #feature extraction:
    ppg_features_dict = PPG_features(data)
    ppg_features = list(ppg_features_dict.values())
    if (ppg_features.count(np.nan) or (ppg_features.count(0)>2) or ((np.sum(ppg_features)/(10e10))>1)):
        raise ValueError("Returned error while extracting features")
    Sample = ppg_features.copy()
    Sample.insert(0,data.loc[0].timestamp)
    
    if not((filespath / ("Sample_"+user_id+".csv")).exists()):
        f_list = list(ppg_features_dict.keys())
        f_list.insert(0,'timestamp')
        f_list.append('triggered')
        f_list.append('filename')
        with open(filespath / ('Sample_'+user_id+'.csv'), 'a', newline='') as file:
            file_writer = csv.writer(file, delimiter=',')
            file_writer.writerow(f_list)
            
#    with open(filespath / ('Sample_'+user_id+'.csv'), 'a', newline='') as file:
#        file_writer = csv.writer(file, delimiter=',')
#        file_writer.writerow(Sample)

    SS = pd.read_csv(filespath / ('Sample_'+user_id+'.csv'))
    if SS.shape[1]==16:
        SS['realtime'] = [-1]*SS.shape[0]
        SS['sleep'] = [-1]*SS.shape[0]
        SS.to_csv(filespath / ('Sample_'+user_id+'.csv'), sep = ',', index = False)
        

    with open(filespath / ('Sample_'+user_id+'.csv')) as file:
        for sample_count, _ in enumerate(file):
            pass
    
    t = False    #TRIGGER SIGNAL, OUTPUT, set to False as default

    if sample_count<0:
        raise ValueError("sample_count should be a non-negative integer")

    elif not(sample_count%100) and sample_count:
        stored_data = np.genfromtxt(filespath / ('Sample_'+user_id+'.csv'),delimiter=',')[1:,1:-4]

        Mean = stored_data.mean(axis=0)
        STD = stored_data.std(axis=0)
        bndrs = np.array((Mean-STD/2, Mean+STD/2)).T
        density = np.zeros(([bndrs.shape[1]+1]*(bndrs.shape[0])))
        
        for row in stored_data:
            index = Sample_Locator(row, bndrs)
            density[tuple(index)]+=1
        np.save(filespath / ('density_'+user_id), density)
        np.savetxt(filespath / ('bndrs_'+user_id+'.csv'), bndrs, delimiter=',')


    elif sample_count>100 and realtime and not(rest_time): #and not(sleep) 
        density = np.load(filespath / ('density_'+user_id+'.npy'))
        bndrs = np.genfromtxt(filespath / ('bndrs_'+user_id+'.csv'), delimiter=',')
        index= Sample_Locator(ppg_features, bndrs)
        d_cal = density[tuple(index)]/density.max()
        d_cal= max(d_cal, 0.15)
        
        eps = np.random.random()
        if eps<d_cal:
            t = True
    
    Sample.append(int(t))
    #Sample.append(str(datapath)[-40:])
    locs = str(datapath).find('data_uniterct')
    Sample.append(str(datapath)[locs:])
    Sample.append(int(realtime))
    Sample.append(int(sleep))
    
    with open(filespath / ('Sample_'+user_id+'.csv'), 'a', newline='') as file:
        file_writer = csv.writer(file, delimiter=',')
        file_writer.writerow(Sample)

    return t

#%%
if __name__ == "__main__":
    
    filepath = Path(r'D:\test111\raw_data')
    dir1 = filepath.parents[0] / 'processed'
    dir1.mkdir(exist_ok = True)
    files = os.listdir(filepath)
    user_id = 'uniterct729'
    
    try:
        os.remove(dir1 / ("Sample_"+user_id+".csv"))
        os.remove(dir1 / ("density_"+user_id+".npy"))
        os.remove(dir1 / ("bndrs_"+user_id+".csv"))
    except:
        pass
        
    files = [files[i] for i in range(len(files)) if (files[i][-4:]=='.csv')]# and (files[i][5:-24]==user_id))]
        
    q=0; p=0; r = 0; s=0
    start = datetime.now()

    for f in files:
        if q%1000==0:
            print(q)

        user_id = f[f.find('uniterct') : f.find('uniterct')+11]
        q+=1
        try:
            trig = main(datapath=filepath / f, filespath= dir1, user_id=user_id)
            if trig:
                r+=1
                print(f, trig)
            

        except ValueError:
            p+=1
            pass
        except:
            s+=1

    print("\n \n", q-p-s, "out of", q, "samples were analyzed.", p+s, "were corrupted or invalid length")
    print("number of Triggers:", r)
    print("run time:  ", datetime.now()-start)
