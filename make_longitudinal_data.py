"""
RITE
Copyright 2023-present NAVER Cloud Corp.
MIT License
"""

import warnings
import os
import sys
import json

import pandas as pd
import numpy as np
import joblib

from tqdm import tqdm

warnings.filterwarnings("ignore")

def load_data(filepath):
    '''
    Load and process glucose
    '''
    glucose = pd.read_csv(
        os.path.join(filepath, 'processed', 'glucose.csv')
        , index_col=0)
    glucose.rename(
        columns={
            'TIME': 'GLCTIMER',
            'VALUE': 'GLC',
        }, inplace=True)
    glucose.drop_duplicates([
        'SUBJECT_ID', 'HADM_ID', 'GLCTIMER', 'GLC', 'GLCSOURCE'
        ], inplace=True)
    glucose.sort_values(by=[
        'SUBJECT_ID', 'HADM_ID', 'GLCTIMER', 'GLCSOURCE', 'GLC'
        ], inplace=True)
    glucose = glucose.groupby([
        'SUBJECT_ID', 'HADM_ID', 'GLCTIMER'
        ]).agg('first').reset_index()

    # load icu stay information
    icu_stay = pd.read_csv(os.path.join(filepath, 'ICUSTAYS.csv'))

    # prepare time information
    icu_stay['INTIME'] = pd.to_datetime(icu_stay['INTIME'], utc=True)
    icu_stay['OUTTIME'] = pd.to_datetime(icu_stay['OUTTIME'], utc=True)
    glucose['GLCTIMER'] = pd.to_datetime(glucose['GLCTIMER'], utc=True)

    # add admission information
    icu_stay.sort_values(by=[
        'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID'
        ], inplace=True)
    icu_stay['first_ICU_stay'] = np.where(
        ~icu_stay.duplicated(subset=['HADM_ID'], keep=False),
        1, 0
        )
    icu_stay.loc[
        icu_stay.groupby(['SUBJECT_ID', 'HADM_ID', 'INTIME']).head(1).index, 'first_ICU_stay'
        ] = 1

    glucose = glucose.merge(icu_stay, on=['SUBJECT_ID', 'HADM_ID'], how='left')

    # reject measurements occur before or after ICU stay
    glucose = glucose[(
            glucose['GLCTIMER'] > glucose['INTIME']
        ) & (
            glucose['GLCTIMER'] < glucose['OUTTIME']
        )]

    # reassign row ID
    glucose = glucose.drop(columns=[
        'ROW_ID'
        ]).reset_index(drop=True).reset_index().rename(
            columns={
                'index': 'GLC_ROW_ID'
            })
    return glucose

def obtain_reserve_data_cdt(data, percentile=0.5, two_sided=True):
    '''
    Find data for reserving
    '''
    if two_sided:
        if 100 - percentile > 50:
            llim = percentile
            ulim = 100 - percentile
        else:
            ulim = percentile
            llim = 100 - percentile

        null_cdt = pd.isnull(data)
        llim = np.percentile(np.squeeze(data[~null_cdt]), llim)
        ulim = np.percentile(np.squeeze(data[~null_cdt]), ulim)

        data_cdt = np.array([True]*len(data))
        data_cdt[~null_cdt] = (
                data[~null_cdt] >= llim
            ) & (
                data[~null_cdt] < ulim
            )
        return data_cdt
    else:
        raise NotImplementedError

def reject_outlier(glc_ins, glc_glc):
    '''
    Reject outliers based on percentile values
    '''
    # intergrate dataframes
    integ_df = pd.concat((glc_ins, glc_glc), axis=0)

    # reject outlier values of height
    reserve_cdt = obtain_reserve_data_cdt(
        integ_df['height'].values, percentile=1
        )
    # reject outlier values of weight
    reserve_cdt = reserve_cdt & obtain_reserve_data_cdt(
        integ_df['weight'].values, percentile=1
        )
    # reject outlier values of age
    reserve_cdt = reserve_cdt & (
            integ_df['AGE'] >= 20
        ) & (
            integ_df['AGE'] < 120
        )
    integ_df = integ_df[reserve_cdt]

    # revert to separated dataframes
    glc_ins_id = glc_ins['GLC_ROW_ID'].to_list()
    glc_glc_id = glc_glc['GLC_ROW_ID'].to_list()

    glc_ins = integ_df[
            integ_df['GLC_ROW_ID'].isin(glc_ins_id)
        ][glc_ins.keys()]
    glc_glc = integ_df[
            integ_df['GLC_ROW_ID'].isin(glc_glc_id)
        ][glc_glc.keys()]
    return glc_ins, glc_glc

def convert_timestamp(x):
    '''
    Convert timestamps to string;
    Return the input value when the conversion is not adoptable
    '''
    try:
        return x.strftime('%Y-%m-%d %H:%m:%S')
    except AttributeError:
        return x

def aggregate_data(df, glucose):
    '''
    Aggregate longitudinal glucose measurements
    '''
    # prepare the dataframe
    df.sort_values(by=[
            'SUBJECT_ID', 'HADM_ID', 'GLCTIMER'
        ], inplace=True, ignore_index=True)
    df['GLCTIMER'] = pd.to_datetime(df['GLCTIMER'], utc=True)

    agg_glc = []
    for _, data in tqdm(
        df.groupby([
                'SUBJECT_ID', 'HADM_ID'
            ])['GLC'].count().reset_index().iterrows()
            ):
        # find corresponding glucose records
        glc = glucose[(
                glucose['SUBJECT_ID'] == data['SUBJECT_ID']
            ) & (
                glucose['HADM_ID'] == data['HADM_ID']
            )]

        # align columns
        for k in df.keys():
            if not k in glc.keys():
                glc.loc[:, k] = [np.nan]*len(glc)
        glc = glc[df.keys()]

        # merge glucose records and treatment events
        glc = pd.merge(
            glc,
            df[(
                    df['SUBJECT_ID'] == data['SUBJECT_ID']
                ) & (
                    df['HADM_ID'] == data['HADM_ID']
                )],
            on=[
                'SUBJECT_ID', 'HADM_ID', 'GLC_ROW_ID', 'GLC', 'GLCTIMER', 'GLCSOURCE'
                ],
            how='outer',
            suffixes=['_x', '']
            )
        glc.sort_values(by=['GLCTIMER'], inplace=True)
        agg_glc.append(glc[df.keys()])
    return agg_glc

def save_data(data, filename):
    if filename.endswith('.pkl'):
        joblib.dump(data, filename)
    elif filename.endswith('.json'):
        data_dict = {}
        data_dict['key'] = list(data[0].keys())
        for d in tqdm(data, total=len(data)):
            for k in d.keys():
                if k.endswith('TIMER'):
                    d[k] = d[k].apply(convert_timestamp)

            subject_id = str(
                    int(d['SUBJECT_ID'].iloc[0])
                ) + '_' + str(
                    int(d['HADM_ID'].iloc[0])
                )
            data_dict[subject_id] = d.values.tolist()

        with open(filename, 'w') as f:
            json.dump(data_dict, f)
    else:
        raise NotImplementedError

def limit_sequence(agg_data, limit=30):
    '''
    (Not used)
    Cut out sequences of glucose records to have the maximum length of 'limit'
    '''
    for i, data in tqdm(enumerate(agg_data)):
        if len(data) > limit:
            ins_idx = np.where(~pd.isnull(data['INSULIN_INPUT']))[0][-1]
            if ins_idx > 0:
                data = data.iloc[:ins_idx+2]
            if len(data) > limit:
                agg_data[i] = data.iloc[:limit]
            else:
                agg_data[i] = data.copy()
    return agg_data

def reject_treatment_subject(df):
    '''
    Exclude subjects with insulin treatment experience from the given data frame
    '''
    ins_sub = pd.read_csv('glc_ins.csv')['SUBJECT_ID'].unique()
    ins_sub = np.append(
        ins_sub,
        pd.read_csv('insulin.csv')['SUBJECT_ID'].unique()
        )
    return df[~df['SUBJECT_ID'].isin(ins_sub)]

try:
    data_path = sys.argv[1]
except IndexError:
    data_path = '../mimic-iii/mimic-iii-clinical-database-1.4/'
save_path = '../data/'
os.makedirs(save_path, exist_ok=True)

# define filenames
ins_filename = 'glc_ins_longitudinal_wfeature.json'
glc_filename = 'glc_glc_longitudinal_wfeature.json'

glc_ins_filename = 'glc_ins_wfeature.csv'
glc_glc_filename = 'glc_glc_wfeature.csv'

# load data
glc_ins_df = pd.read_csv(glc_ins_filename)
glc_glc_df = pd.read_csv(glc_glc_filename)
glucose_df = load_data(data_path)

# reject outliers
glc_ins_df, glc_glc_df = reject_outlier(glc_ins_df, glc_glc_df)
# save filtered dataframes
glc_ins_df.to_csv(
    os.path.join(
        save_path,
        glc_ins_filename.replace('.csv', '_filtered.csv')
    ), index=False)
glc_glc_df.to_csv(
    os.path.join(
        save_path,
        glc_glc_filename.replace('.csv', '_filtered.csv')
    ), index=False)

# aggregate treatment group data
glc_ins_df = aggregate_data(glc_ins_df, glucose_df)
save_data(glc_ins_df, os.path.join(save_path, ins_filename))

# aggregate control group data
glc_glc_df = reject_treatment_subject(glc_glc_df)
glc_glc_df = aggregate_data(glc_glc_df, glucose_df)
save_data(glc_glc_df, os.path.join(save_path, glc_filename))
