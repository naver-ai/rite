"""
RITE
Copyright 2023-present NAVER Cloud Corp.
MIT License
"""

import os
import json
import random
import numpy as np
import pandas as pd

from tqdm import tqdm

def data_check_glc(data):
    treat = 0
    loc = len(data)-1
    while (treat == 0) & (loc > 0):
        if pd.isna(data[loc][7]):
            loc -= 1
        else:
            treat = 1
    return data[0:loc+1]

def data_check_ins(data):
    treat = 0
    loc = len(data)-1
    while treat == 0 and loc>0:
        if (pd.isna(data[loc][6]) == 1) | (pd.isna(data[loc][9])):
            loc -= 1
        else:
            treat = 1
    return data[0:loc+1]

def static_check_glc(data, timestamps):
    check = np.zeros((5,))
    for i in range(len(data)-timestamps, len(data)):
        if not pd.isna(data[i][14]):
            check[0] = 1.
        if not pd.isna(data[i][16]):
            check[1] = 1.
        if not pd.isna(data[i][18]):
            check[2] = 1.
        if not pd.isna(data[i][19]):
            check[3] = 1.
        if not pd.isna(data[i][21]):
            check[4] = 1.
        return int(sum(check) == 5.)

def static_check_ins(data, timestamps):
    check = np.zeros((5,))
    for i in range(len(data)-timestamps, len(data)):
        if not pd.isna(data[i][21]):
            check[0] = 1.
        if not pd.isna(data[i][23]):
            check[1] = 1.
        if not pd.isna(data[i][25]):
            check[2] = 1.
        if not pd.isna(data[i][26]):
            check[3] = 1.
        if not pd.isna(data[i][28]):
            check[4] = 1.
        return int(sum(check) == 5.)

def get_eth(eth, eth_dic={
    'WHITE': 1.0,
    'BLACK/AFRICAN AMERICAN': 2.0,
    'PATIENT DECLINED TO ANSWER': 3.0,
    'OTHER': 4.0,
    'ASIAN - VIETNAMESE': 5.0,
    'PORTUGUESE': 6.0,
    'HISPANIC OR LATINO': 7.0,
    'UNKNOWN/NOT SPECIFIED': 8.0,
    'ASIAN - OTHER': 9.0,
    'ASIAN': 10.0,
    'WHITE - OTHER EUROPEAN': 11.0,
    'WHITE - RUSSIAN': 12.0,
    'UNABLE TO OBTAIN': 13.0,
    'ASIAN - ASIAN INDIAN': 14.0,
    'ASIAN - CHINESE': 15.0,
    'HISPANIC/LATINO - CUBAN': 16.0,
    'WHITE - EASTERN EUROPEAN': 17.0,
    'MIDDLE EASTERN': 18.0,
    'HISPANIC/LATINO - DOMINICAN': 19.0,
    'HISPANIC/LATINO - PUERTO RICAN': 20.0,
    'ASIAN - JAPANESE': 21.0,
    'BLACK/AFRICAN': 22.0,
    'BLACK/CAPE VERDEAN': 23.0,
    'HISPANIC/LATINO - GUATEMALAN': 24.0,
    'BLACK/HAITIAN': 25.0,
    'HISPANIC/LATINO - MEXICAN': 26.0,
    'MULTI RACE ETHNICITY': 27.0,
    'ASIAN - CAMBODIAN': 28.0,
    'ASIAN - KOREAN': 29.0,
    'WHITE - BRAZILIAN': 30.0,
    'ASIAN - FILIPINO': 31.0,
    'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER': 32.0,
    'HISPANIC/LATINO - SALVADORAN': 33.0,
    'HISPANIC/LATINO - COLOMBIAN': 34.0,
    'CARIBBEAN ISLAND': 35.0,
    'ASIAN - THAI': 36.0,
    'AMERICAN INDIAN/ALASKA NATIVE': 37.0,
    'SOUTH AMERICAN': 38.0,
    'HISPANIC/LATINO - CENTRAL AMERICAN (OTHER)': 39.0,
    'AMERICAN INDIAN/ALASKA NATIVE FEDERALLY RECOGNIZED TRIBE': 40.0
    }):
    if pd.isna(eth):
        return 0.
    else:
        return eth_dic[eth]/100.

def get_time(timegap):
    if pd.isna(timegap):
        return 0.
    else:
        temp = timegap
        temp = temp.split()
        temp1 = temp[0].split('-')
        year = float(temp1[0])*24*365
        month = float(temp1[1])*24*30
        day = float(temp1[2])*24
        temp2 = temp[1].split('+')[0]
        temp2 = temp2.split(':')
        hours = float(temp2[0])
        mins = float(temp2[1])/60
        secs = float(temp2[2])/3600.
        return float(year+month+day+hours+mins+secs)

def load_data(path_to_data):
    #source data load
    with open(
        os.path.join(path_to_data, "glc_glc_longitudinal_wfeature.json")
        , "r") as st_json:
        ours_glc = json.load(st_json)

    with open(
        os.path.join(path_to_data, "glc_ins_longitudinal_wfeature.json")
        , "r") as st_json:
        ours_ins = json.load(st_json)
    return ours_glc, ours_ins

def obtain_id(ours_glc, ours_ins):
    ins_ids = list(ours_ins.keys())
    ins_ids.remove('key')
    glc_ids = list(ours_glc.keys())
    glc_ids.remove('key')

    random.seed(42)

    # shuffle
    random.shuffle(ins_ids)
    random.shuffle(glc_ids)
    return glc_ids, ins_ids

def preproc(ours_glc, glc_ids, ours_ins, ins_ids, len_seq=10):
    # preprocess for getting data without nan data and with enough timestamps
    # treatment at the final timestamp in treatment group = 1
    glc_pre = []
    ins_pre = []
    for i in glc_ids:
        ours_glc[i] = data_check_glc(ours_glc[i])
        if len(ours_glc[i]) >= len_seq:
            glc_pre.append(i)
    for i in ins_ids:
        ours_ins[i] = data_check_ins(ours_ins[i])
        if len(ours_ins[i]) >= len_seq:
            ins_pre.append(i)
    return glc_pre, ins_pre

def find_last_index(target_folder):
    filelist = os.listdir(target_folder)
    last_idx = 0
    for f in filelist:
        try:
            idx = int(f.split('.')[0])
            if idx > last_idx:
                last_idx = idx + 1
        except ValueError:
            continue
    return last_idx

def extract_feature(
    data,
    save_path='../data/processed',
    timestamps=10,
    glc_norm=1000,
    time_norm=100.,
    norm=100.
    ):
    '''
    Extract features from data referring the column indices in below
    Treatment group             Control group
    0:  'SUBJECT_ID',            'SUBJECT_ID',
    1:  'HADM_ID',               'HADM_ID',
    2:  'GLC_ROW_ID',            'GLC_ROW_ID',
    3:  'GLC',                   'GLC',
    4:  'GLCTIMER',              'GLCTIMER',
    5:  'INSULIN_ROW_ID',        'INSULIN_ROW_ID',
    6:  'INSULIN_TIMER',         'FU_GLC_ROW_ID',
    7:  'INSULIN_TIMER_GAP',     'FU_GLC',
    8:  'FU_GLC_ROW_ID',         'FU_GLCTIMER',
    9:  'FU_GLC',                'FU_GLCTIMER_GAP',
    10: 'FU_GLCTIMER',           'GLC_ITEMID',
    11: 'FU_GLCTIMER_GAP',       'GLCSOURCE',
    12: 'GLC_ITEMID',            'FU_GLC_ITEMID',
    13: 'GLCSOURCE',             'FU_GLCSOURCE'
    14: 'FU_GLC_ITEMID',         'height',
    15: 'FU_GLCSOURCE',          'ADMISSION_TYPE',
    16: 'INSULIN_INPUT',         'ETHNICITY',
    17: 'INSULIN_INPUT_HRS',     'DIAGNOSIS',
    18: 'INSULINTYPE',           'GENDER',
    19: 'INSULIN_EVENT',         'AGE',
    20: 'INSULIN_INFXSTOP'       'weight'
    21: 'height',                
    22: 'ADMISSION_TYPE',
    23: 'ETHNICITY',
    24: 'DIAGNOSIS',
    25: 'GENDER',
    26: 'AGE',
    27: 'weight'
    '''
    for i, d in tqdm(enumerate(data)):
        a = np.zeros((timestamps,))
        s = np.zeros((4,))
        x = np.zeros((timestamps, 2))
        time_0 = get_time(d[0][4])
        if len(d[0]) < 25: # control group
            y = np.array([d[timestamps-1][7]])/glc_norm
            for j in range(timestamps): # for glc
                a[j] = 0.
                x[j][0] = d[j][3]/glc_norm # glc
                x[j][1] = max(get_time(d[j][4])-time_0, 0)/time_norm# glc timer
                if not pd.isna(d[j][19]):
                    s[0] = d[j][19]/norm # age
                if not pd.isna(d[j][18]):
                    s[1] = d[j][18]+1. # gender
                if not pd.isna(d[j][20]) and not pd.isna(d[j][14]):
                    s[2] = d[j][20]/(d[j][14]**2)*10000./norm # BMI
                if not pd.isna(d[j][16]):
                    s[3] = get_eth(d[j][16]) # ETHNICITY
        else: # treatment group
            y = np.array([d[timestamps-1][9]])/glc_norm
            for j in range(timestamps): # for ins
                a[j] = 1.-float(pd.isna(d[j][16]))
                x[j][0] = d[j][3]/glc_norm # glc
                x[j][1] = max(get_time(d[j][4])-time_0, 0)/time_norm # glc timer
                if not pd.isna(d[j][26]):
                    s[0] = d[j][26]/norm # age
                if not pd.isna(d[j][25]):
                    s[1] = d[j][25]+1. # gender
                if not pd.isna(d[j][27]) and not pd.isna(d[j][21]):
                    s[2] = d[j][27]/(d[j][21]**2)*10000./norm # BMI
                if not pd.isna(d[j][23]):
                    s[3] = get_eth(d[j][23]) # ETHNICITY

        # save_data
        os.makedirs(save_path, exist_ok=True)
        idx = find_last_index(save_path)
        np.save(os.path.join(save_path, f'{str(idx + i)}.a.npy'), a)
        np.save(os.path.join(save_path, f'{str(idx + i)}.static.npy'), s)
        np.save(os.path.join(save_path, f'{str(idx + i)}.x.npy'), x)
        np.save(os.path.join(save_path, f'{str(idx + i)}.y.npy'), y)

data_path = '../data/'
glc, ins = load_data(data_path)
glc_id, ins_id = obtain_id(glc, ins)
glc_id, ins_id = preproc(glc, glc_id, ins, ins_id)

# split train/test/valid IDs
ins_train = ins_id[0:5000]
ins_test = ins_id[5000:5750]
ins_valid = ins_id[5750:5950]
glc_train = glc_id[0:5000]
glc_test = glc_id[5000:5750]
glc_valid = glc_id[5750:5950]

# collect data
train_data = []
test_data = []
valid_data = []
timestamp=10
for ids in ins_train:
    train_data.append(ins[ids][len(ins[ids])-timestamp:len(ins[ids])])
for ids in ins_test:
    test_data.append(ins[ids][len(ins[ids])-timestamp:len(ins[ids])])
for ids in ins_valid:
    valid_data.append(ins[ids][len(ins[ids])-timestamp:len(ins[ids])])

for ids in glc_train:
    train_data.append(glc[ids][len(glc[ids])-timestamp:len(glc[ids])])
for ids in glc_test:
    test_data.append(glc[ids][len(glc[ids])-timestamp:len(glc[ids])])
for ids in glc_valid:
    valid_data.append(glc[ids][len(glc[ids])-timestamp:len(glc[ids])])

random.shuffle(train_data)
random.shuffle(test_data)
random.shuffle(valid_data)

# extract feature from train, valid, and test data
extract_feature(train_data, timestamps=timestamp)
extract_feature(valid_data, timestamps=timestamp)
extract_feature(test_data, timestamps=timestamp)

# save data index
split_list = [1]*len(train_data) + [0]*len(test_data) + [2]*len(valid_data)

df = pd.DataFrame(split_list)
df.to_csv('../data/train_test_split_mimic_glcins.csv', index=False)
