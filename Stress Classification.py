# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 10:46:51 2021
@author: Ali

************************************
Stress Classification Subjective
************************************
"""

import csv
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier as NN
from sklearn.neighbors import KNeighborsClassifier as KNN
import os
import heartpy as hp
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


#%% This block reads two sets of files: the extracted features from each sample, and the entire list of labels (downloaded from Dashboard)
# The output (labeled_data) is a dataframe of labeled data: For each label if there is any sample within 16 minutes around it, the label is assigned to that sample.

dir_figs = Path(r'D:\UCI\6th year\Winter\Unite\Code\Figures')
dir_samples1 = Path(r'D:\UCI\Unite\Unite_RCT\Source Data\processed\processed 2 minutes offline')
dir_samples2 = Path(r'D:\UCI\Unite\Unite_RCT\Source Data\processed\processed 2 minutes offline - from 5 minutes')
dir_labels = Path(r'D:\UCI\Unite\Unite_RCT\Source Data\Labels')

files = os.listdir(dir_samples1)
files = [f for f in files if f[:7]=='Sample_']

dataset = pd.read_csv(dir_samples1 / files[0])
dataset.insert(0,'user',[files[0][7:-4]]*len(dataset))

for f in files[1:]:
    data = pd.read_csv(dir_samples1 / f)
    data.insert(0,'user',[f[7:-4]]*len(data))
    dataset = dataset.append(data, ignore_index=True)

files = os.listdir(dir_samples2)
files = [f for f in files if f[:7]=='Sample_']

for f in files:
    data = pd.read_csv(dir_samples2 / f)
    data.insert(0,'user',[f[7:-4]]*len(data))
    dataset = dataset.append(data, ignore_index=True)

users = np.unique(dataset.user)
hrvs = data.columns[2:-4]

stress_labels = ['not at all', 'a little bit', 'some', 'a lot', 'extremely']
labels = pd.read_csv(dir_labels / 'labels.csv', sep = ',')
print(labels.shape)
labels = labels[labels['name'].isin(users)]
print(labels.shape)
labels = labels[labels['data.stressed']!='undefined']
print(labels.shape)
labels = labels[labels['data.stressed'].isin(stress_labels)]
print(labels.shape)

#labels['data.stressed'] = labels['data.stressed'].replace({'not at all': 0, 'a little bit': 0,
#                                                          'some':1, 'a lot':1, 'extremely':1})

labeled_data = pd.DataFrame(columns=dataset.columns)
stress = []
delay = 16 #60*24*3 #minutes

labels['data.stressed_last_modify'] = pd.to_numeric(labels['data.stressed_last_modify'])
for i in range(len(labels)):
    dataset_user = dataset[dataset['user']==labels.name.iloc[i]].copy()
    dataset_user.reset_index(drop=True, inplace=True)
    
    td = -dataset_user['timestamp']+labels['data.stressed_last_modify'].iloc[i]
    td = td.abs()
    if any((td<(delay*6e4))):
        labeled_data = labeled_data.append(dataset_user.loc[td.argmin()], ignore_index=True)
        stress.append(labels['data.stressed'].iloc[i])
        
        
labeled_data = labeled_data[['user']+list(hrvs)]
labeled_data['stress'] = stress

print(np.unique(labeled_data['stress'], return_counts=True))
print(np.unique(labeled_data['user'], return_counts=True))


#%% Objective (all users together), maping all stress levels to binary

stress_conv_dic = {'not at all': 0, 'a little bit': 0, 'some':1, 'a lot':1, 'extremely':1}

ldata = labeled_data.copy()
ldata.replace(stress_conv_dic, inplace=True)
print(np.unique(ldata['stress'], return_counts=True))
#labeled_data_2_class['stress'] = labeled_data_2_class['stress'].replace(c,1)


X = ldata[ldata.columns[1:-1]].values
y = ldata['stress'].values

over = SMOTE(sampling_strategy=0.6)
under = RandomUnderSampler(sampling_strategy=1)
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)
X, y = pipeline.fit_resample(X, y)
print(np.unique(y, return_counts=True))


scaler = StandardScaler()
X = scaler.fit_transform(X)

X, y = shuffle(X, y)
scoring = ['f1', 'f1_micro', 'f1_macro', 
           'precision_macro', 'precision_micro', 'precision',
           'recall_macro', 'recall_micro', 'recall']
#*******************************************uncomment one Classifier
#Classifier = NN(hidden_layer_sizes= (25, 15), max_iter = 500)
#Classifier = SVC(kernel = 'rbf', class_weight = 'balanced')
#Classifier = RF(class_weight = 'balanced', max_depth = 7)
Classifier = XGBClassifier(max_depth=7)
#Classifier = KNN(n_neighbors = 9)


scorer = cross_validate(Classifier, X, y, cv=5, scoring = scoring)

#print(f"Micro: {scorer['test_f1_micro'].mean():.2f} \u00B1 {scorer['test_f1_micro'].std():.2f}")
#print(f"Macro: {scorer['test_f1_macro'].mean():.2f} \u00B1 {scorer['test_f1_macro'].std():.2f}")
print(f"F1:  {scorer['test_f1'].mean():.2f} \u00B1 {scorer['test_f1'].std():.2f}")
beta = 0.5
f05 = (1+beta**2)/(1/scorer['test_precision']+beta**2/scorer['test_recall'])
print(f"F05: {f05.mean():.2f} \u00B1 {f05.std():.2f}")

#%% Objective (all users together), each class vs baseline

ldata = labeled_data.copy() 
stress_conv_dic = {'not at all': 0, 'a little bit': 1, 'some':1, 'a lot':1, 'extremely':1}
ldata.replace({'extremely': 'a lot'}, inplace=True)

for c in list(stress_conv_dic.keys())[1:-1]:
    ldata_2 = ldata.copy() 
    ldata_2 = ldata_2[ldata_2['stress'].isin(['not at all',c])]
    ldata_2.replace(stress_conv_dic, inplace=True)
    print(np.unique(ldata_2['stress'], return_counts=True))

    X = ldata_2[ldata_2.columns[1:-1]].values
    y = ldata_2['stress'].values
    
    over = SMOTE(sampling_strategy=0.6)
    under = RandomUnderSampler(sampling_strategy=1)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    X, y = pipeline.fit_resample(X, y)
    print(np.unique(y, return_counts=True))
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X, y = shuffle(X, y)
    scoring = ['f1', 'f1_micro', 'f1_macro', 
               'precision_macro', 'precision_micro', 'precision',
               'recall_macro', 'recall_micro', 'recall']
    
    #*******************************************uncomment one Classifier
    #Classifier = NN(hidden_layer_sizes= (50, 30), max_iter = 500)
    #Classifier = SVC(kernel = 'rbf', class_weight = 'balanced')
    Classifier = RF(class_weight = 'balanced', max_depth = 7)
    #Classifier = XGBClassifier(max_depth=7)
    #Classifier = KNN(n_neighbors = 1)
    scorer = cross_validate(Classifier, X, y, cv=5, scoring = scoring)
    
    #print(f"Micro: {scorer['test_f1_micro'].mean():.2f} \u00B1 {scorer['test_f1_micro'].std():.2f}")
    #print(f"Macro: {scorer['test_f1_macro'].mean():.2f} \u00B1 {scorer['test_f1_macro'].std():.2f}")
    print(f"F1: {scorer['test_f1'].mean():.2f} \u00B1 {scorer['test_f1'].std():.2f}")


    
#%% Training on data from other users, test on outside subject
stress_conv_dic = {'not at all': 0, 'a little bit': 0, 'some':1, 'a lot':1, 'extremely':1}

ldata = labeled_data.copy()
ldata.replace(stress_conv_dic, inplace=True)
print(np.unique(ldata['stress'], return_counts=True))

precision = []
recall = []
f1 = []
users = ['uniterct206', 'uniterct446', 'uniterct470',
       'uniterct552', 'uniterct761'] # 'uniterct729',

for u in users:   
    print(u)
    X_tr = ldata[ldata.user!=u]
    X_te = ldata[ldata.user==u]
    y_tr = X_tr['stress'].values
    y_te = X_te['stress'].values
    X_tr = X_tr[hrvs].values
    X_te = X_te[hrvs].values
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    
    over = SMOTE(sampling_strategy=0.6)
    under = RandomUnderSampler(sampling_strategy=1)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    X_tr, y_tr = pipeline.fit_resample(X_tr, y_tr)
    print(np.unique(y_tr, return_counts=True))
    print(np.unique(y_te, return_counts=True))
    

    Classifier = XGBClassifier(max_depth=7)
    Classifier.fit(X_tr,y_tr)
    
    scores = score(y_te, Classifier.predict(X_te))
    precision.append(scores[0])
    recall.append(scores[1])
    f1.append(scores[2])


f1 = np.array(f1)


#%% Fine tune on data from subjects
precision_s = []
recall_s = []
f1_s = []
ls = []
for u in users:
    X_tr = labeled_data[labeled_data.user!=u]
    X_te = labeled_data[labeled_data.user==u]
    
    l = X_te.shape[0]
    
    if l<51:
        continue
    ls.append(l)
    X_tr = X_tr.append(X_te[:int(l/2)])
    X_te = X_te[int(l/2):]
    
    y_tr = X_tr['stress']
    y_te = X_te['stress']
    X_tr = X_tr[hrvs]
    X_te = X_te[hrvs]
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    C_SVM = SVC(kernel = 'rbf', class_weight = 'balanced')
    C_SVM.fit(X_tr,y_tr)
    
    scores = score(y_te, C_SVM.predict(X_te))
    precision_s.append(scores[0])
    recall_s.append(scores[1])
    f1_s.append(scores[2])

f1_s = np.array(f1_s)


i=0
plt.plot(f1[:,i], color='red')
plt.plot(f1_s[:,i], color='blue')


#%% Progress over number of samples

stress_conv_dic = {'not at all': 0, 'a little bit': 0, 'some':1, 'a lot':1, 'extremely':1}

ldata = labeled_data.copy()
ldata.replace(stress_conv_dic, inplace=True)
print(np.unique(ldata['stress'], return_counts=True))

X = ldata[ldata.columns[1:-1]]
y = ldata['stress']
X = StandardScaler().fit_transform(X)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify = y)

over = SMOTE(sampling_strategy=0.6)
under = RandomUnderSampler(sampling_strategy=1)
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)
X_tr, y_tr = pipeline.fit_resample(X_tr, y_tr)
print(np.unique(y_tr, return_counts=True))
print(np.unique(y_te, return_counts=True))

f1_mean = []
f1_std = []

precision = []
recall = []
f1 = []
for i in range(1,8):
        

    X_tr_sub = X_tr[:i*100]
    y_tr_sub = y_tr[:i*100]
    
    Classifier = XGBClassifier(max_depth = 7)
    Classifier.fit(X_tr_sub,y_tr_sub)
    
    #print(score(y_te.values, C_SVM.predict(X_te)))
    scores = score(y_te, Classifier.predict(X_te), average='macro')
    precision.append(scores[0])
    recall.append(scores[1])
    f1.append(scores[2])
#        
#    f1_mean.append(np.mean(f1, axis=0))
#    f1_std.append(np.std(f1, axis=0))
f1 = np.array(f1)

#%%
f1_mean = np.array(f1_mean)
f1_std = np.array(f1_std)
print(f1_mean)

c = 1
plt.figure(figsize=(8,5))
plt.plot(np.arange(2,12)*50, f1_mean[1:,c])
plt.fill_between(np.arange(2,12)*50, f1_mean[1:,c]-f1_std[1:,c],f1_mean[1:,c]+f1_std[1:,c], color='blue', alpha=0.1)
#plt.plot(np.arange(2,12)*50, f1_mean[1:,0]+f1_std[1:,0], color='red')
plt.xlabel("number of samples used")
plt.ylabel("F1 score for test data")
#plt.savefig(dir_figs / 'Prediction progress over number of samples.pdf')
