"""
RITE
Copyright 2023-present NAVER Cloud Corp.
MIT License
"""

import os
import sys

import pandas as pd
import numpy as np

from tqdm import tqdm

try:
    data_path = sys.argv[1]
except IndexError:
    data_path = '../mimic-iii/mimic-iii-clinical-database-1.4/'
os.makedirs(os.path.join(data_path, 'processed'), exist_ok=True)

## --------------------------------------------------------------------- ##
## 1. Chart event
## --------------------------------------------------------------------- ##

chunksize = 100000
chart = pd.DataFrame()
for chunk in tqdm(pd.read_csv(os.path.join(data_path, 'CHARTEVENTS.csv'), chunksize=chunksize)):
    # extract glucose measurements
    chunk = chunk[chunk['ITEMID'].isin([
                            807,    #   Fingerstick Glucose
                            811,    #   Glucose (70-105)
                            1529,   #	Glucose
                            3745,   #	BloodGlucose
                            3744,   #	Blood Glucose
                            225664, #	Glucose finger stick
                            220621, #	Glucose (serum)
                            226537
                        ])]

    # remove glucose measurement records with error
    chunk = chunk[chunk['ERROR'] != 1]

    # replace missing values with valuenum
    chunk['VALUE'] = np.where(pd.isnull(chunk['VALUE']), chunk['VALUENUM'], chunk['VALUE'])

    # convert glucose measurement to numeric
    chunk['VALUE'] = pd.to_numeric(chunk['VALUE'], errors='coerce')

    # reject missing values
    chunk = chunk[~pd.isnull(chunk['VALUE'])]
    chunk = chunk[chunk['VALUE'] > 0]
    chart = pd.concat((chart, chunk), axis=0)

# determine time information
chart['TIME_GAP'] = pd.to_datetime(chart['STORETIME']) - pd.to_datetime(chart['CHARTTIME'])

chart['TIME'] = np.where(
    chart['TIME_GAP'] < pd.Timedelta(0),
    chart['STORETIME'],
    chart['CHARTTIME']
    )
chart.drop(columns=[
    'TIME_GAP', 'STORETIME', 'CHARTTIME'
    ], inplace=True)

# remove duplications
chart.drop_duplicates([
    'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'ITEMID', 'VALUE', 'TIME'
    ], inplace=True)

# filter out inaccurate measurements
inaccurate_cdt_lab_analysis = (
        chart['ITEMID'].isin([3744, 3745, 220621])
    ) & (
        chart['VALUE'] > 1000
        )
inaccurate_cdt_fingerstick = (
        chart['ITEMID'].isin([807, 811, 1529, 225664, 226537])
    ) & (
        chart['VALUE'] > 500
        )
chart.drop(chart[(inaccurate_cdt_lab_analysis) | (inaccurate_cdt_fingerstick)].index, inplace=True)

chart = chart[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'ITEMID', 'TIME', 'VALUE']]
chart.to_csv(os.path.join(data_path, 'processed', 'chart.csv'))

## --------------------------------------------------------------------- ##
## 2. Lab event
## --------------------------------------------------------------------- ##
lab = pd.read_csv(os.path.join(data_path, 'LABEVENTS.csv'))

# extract glucose measurements
lab = lab[lab['ITEMID'].isin([
                        50931, #    GLUCOSE | CHEMISTRY | BLOOD
                        50809  #    GLUCOSE | BLOOD GAS | BLOOD
                    ])]

# replace missing values with valuenum
lab['VALUE'] = np.where(pd.isnull(lab['VALUE']), lab['VALUENUM'], lab['VALUE'])

# convert glucose measurement to numeric
lab['VALUE'] = pd.to_numeric(lab['VALUE'], errors='coerce')

# reject missing values
lab = lab[~pd.isnull(lab['VALUE'])]
lab = lab[lab['VALUE'] > 0]

# determine time information
lab.rename(columns={'CHARTTIME': 'TIME'}, inplace=True)

# remove duplications
lab.drop_duplicates(['SUBJECT_ID', 'HADM_ID', 'ITEMID', 'TIME', 'VALUE'], inplace=True)

# filter out inaccurate measurements
lab = lab[lab['VALUE'] <= 1000]

lab = lab[['SUBJECT_ID', 'HADM_ID', 'ITEMID', 'TIME', 'VALUE']]
lab.to_csv(os.path.join(data_path, 'processed', 'lab.csv'))

## --------------------------------------------------------------------- ##
## 3. Integration
## --------------------------------------------------------------------- ##

glucose = pd.concat((chart[['SUBJECT_ID', 'HADM_ID', 'ITEMID', 'TIME', 'VALUE']], lab), axis=0)
glucose.drop_duplicates(inplace=True)

# assign glucose source
glucose['GLCSOURCE'] = np.where(
    glucose['ITEMID'].isin([3744, 3745, 220621, 50931, 50809]),
    'BLOOD',
    'FINGERSTICK'
    )

glucose.to_csv(os.path.join(data_path, 'processed', 'glucose.csv'))
