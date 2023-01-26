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

def extract_data(filepath, save=True):
    '''
    Extract relevant weight, height, and demographic information
    '''
    weight_df = pd.DataFrame()
    height_df = pd.DataFrame()
    for data in tqdm(pd.read_csv(os.path.join(filepath, 'CHARTEVENTS.csv'), chunksize=500000)):
        # daily weight: 763 224639
        # present weight: 3580 3581 3582
        # admission weight: 226512 226531
        # weight: 3693
        weight_df = pd.concat((
            weight_df,
            data[data['ITEMID'].isin([763, 224639, 3580, 3581, 3582, 226512, 226531, 3693])]
            ), axis=0)

        height_df = pd.concat((
            height_df,
            data[data['ITEMID'].isin([226707, 226730])]
            ), axis=0)

    patient_df = pd.read_csv(os.path.join(filepath, 'PATIENTS.csv'))
    admission_df = pd.read_csv(os.path.join(filepath, 'ADMISSIONS.csv'))
    admission_df = admission_df.merge(
        patient_df[['SUBJECT_ID', 'GENDER', 'DOB']],
        on=['SUBJECT_ID'],
        how='left'
        )

    # gender
    admission_df['GENDER'] = np.where(admission_df['GENDER'] == 'M', 0, 1)

    # age
    admission_df['DOB'] = pd.to_datetime(admission_df['DOB'])
    admission_df['ADMITTIME'] = pd.to_datetime(admission_df['ADMITTIME'])
    admission_df['AGE'] =  admission_df['ADMITTIME'].dt.date - admission_df['DOB'].dt.date
    admission_df['AGE'] = admission_df['AGE'].apply(
        lambda x: np.floor(x.days/365.25)
        )
    admission_df = admission_df[[
        'SUBJECT_ID', 'HADM_ID', 'ADMISSION_TYPE', 'ETHNICITY',
        'DIAGNOSIS', 'GENDER', 'AGE'
        ]]

    if save:
        weight_df.to_csv(
            os.path.join(filepath, 'processed', 'weight.csv')
            , index=False)
        height_df.to_csv(
            os.path.join(filepath, 'processed', 'height.csv')
            , index=False)
        admission_df.to_csv(
            os.path.join(filepath, 'processed', 'demographics.csv')
            , index=False)
    return weight_df, height_df, admission_df

def filter_data(filepath, save=True):
    '''
    Unify the units and filter invalid values out for weight and height data
    '''
    weight_df = pd.read_csv(
        os.path.join(filepath, 'processed', 'weight.csv')
        )
    height_df = pd.read_csv(
        os.path.join(filepath, 'processed', 'height.csv')
        )

    # reject uncertain data
    weight_df = weight_df[weight_df['ERROR'] != 1]
    weight_df = weight_df[~weight_df['VALUEUOM'].isin(['cmH20', 'gms'])]
    weight_df = weight_df[weight_df['VALUE'] > 0]

    height_df = height_df[height_df['ERROR'] != 1]
    height_df = height_df[height_df['VALUE'] > 0]

    # unify units
    weight_df['VALUE_kg'] = np.where(weight_df['VALUEUOM'] == 'kg', weight_df['VALUE'], np.nan)
    for lb_weight_id in [3581, 226531]:
        cdt = weight_df['ITEMID'] == lb_weight_id
        weight_df.loc[cdt, 'VALUE_kg'] = weight_df[cdt]['VALUE'] * 0.45359237
    weight_df = weight_df[~pd.isnull(weight_df['VALUE_kg'])]

    height_df['VALUE_cm'] = np.where(
        height_df['VALUEUOM'] == 'cm',
        height_df['VALUE'],
        height_df['VALUE']*2.54
        )
    height_df = height_df[~pd.isnull(height_df['VALUE_cm'])]

    # aggregate duplicated measurements
    weight_df.sort_values(by=[
        'SUBJECT_ID', 'HADM_ID', 'CHARTTIME'
        ], inplace=True)
    weight_df = weight_df.groupby([
        'SUBJECT_ID', 'HADM_ID', 'CHARTTIME'
        ])['VALUE_kg'].agg(np.mean).reset_index()

    height_df.sort_values(by=[
        'SUBJECT_ID', 'HADM_ID', 'CHARTTIME'
        ], inplace=True)
    height_df = height_df.groupby([
        'SUBJECT_ID', 'HADM_ID', 'CHARTTIME'
        ])['VALUE_cm'].agg(np.mean).reset_index()

    if save:
        weight_df.to_csv(
            os.path.join(filepath, 'processed', 'filtered_weight.csv')
            , index=False)
        height_df.to_csv(
            os.path.join(filepath, 'processed', 'filtered_height.csv')
            , index=False)
    return weight_df, height_df

def merge_dfs(target_dfs, source_dfs, keys):
    '''
    A function to merge multiple combinations of dataframes
    '''
    resulted_df = []
    for source_df in source_dfs:
        for target_df in target_dfs:
            resulted_df.append(
                target_df.merge(source_df, on=keys, how='left')
            )
    return resulted_df

def arrange_columns(target_dfs, operation, info):
    '''
    A function to repeatedly conduct the given column operation for multiple dataframes
    '''
    resulted_df = []
    for target_df in target_dfs:
        if operation == 'rename':
            resulted_df.append(
                target_df.rename(columns={info[0]: info[1]})
            )
        elif operation == 'drop':
            resulted_df.append(
                target_df.drop(info, axis=1)
            )
        elif operation == 'fill':
            target_df[info[0]] = np.where(
                pd.isnull(target_df[info[0]]),
                target_df[info[1]],
                target_df[info[0]]
                )
            resulted_df.append(
                target_df
            )
        else:
            raise NotImplementedError
    return resulted_df

def attach_data(filepath, csv_path, save=True):
    '''
    Attach the additional information to the control and treatment group data
    '''
    # load data
    ins = pd.read_csv(os.path.join(csv_path, 'glc_ins.csv'))
    glc = pd.read_csv(os.path.join(csv_path, 'glc_glc.csv'))

    weight_df = pd.read_csv(
        os.path.join(filepath, 'processed', 'filtered_weight.csv')
        )
    height_df = pd.read_csv(
        os.path.join(filepath, 'processed', 'filtered_height.csv')
        )
    demo_df = pd.read_csv(
        os.path.join(filepath, 'processed', 'demographics.csv')
        )

    # aggregate information
    height_df = height_df.groupby([
        'SUBJECT_ID', 'HADM_ID'
        ])['VALUE_cm'].agg(np.mean).reset_index()
    [ins, glc] = merge_dfs(
        [ins, glc],
        [height_df],
        keys=['SUBJECT_ID', 'HADM_ID']
        )
    [ins, glc] = arrange_columns(
        [ins, glc],
        'rename',
        ['VALUE_cm', 'height']
        )

    [ins, glc] = merge_dfs(
        [ins, glc],
        [demo_df],
        keys=['SUBJECT_ID', 'HADM_ID']
        )

    # attach weight based on day info
    weight_df['ref'] = pd.to_datetime(weight_df['CHARTTIME']).dt.date
    ins['ref'] = pd.to_datetime(ins['GLCTIMER']).dt.date
    glc['ref'] = pd.to_datetime(glc['GLCTIMER']).dt.date

    ref_weight_df = weight_df.groupby([
        'SUBJECT_ID', 'HADM_ID', 'ref'
        ])['VALUE_kg'].agg(np.mean).reset_index()
    [ins, glc] = merge_dfs(
        [ins, glc],
        [ref_weight_df],
        keys=['SUBJECT_ID', 'HADM_ID', 'ref']
        )
    [ins, glc] = arrange_columns(
        [ins, glc],
        'rename',
        ['VALUE_kg', 'weight']
        )

    print(str(
        pd.isnull(ins['weight']).sum()
        ) + '\t weight data are still missing in treatment group')
    print(str(
        pd.isnull(glc['weight']).sum()
        ) + '\t weight data are still missing in controlled group')

    # attach remained weight based on month info
    weight_df['ref'] = pd.to_datetime(
            weight_df['CHARTTIME']
        ).dt.year.astype(str) + '-' + pd.to_datetime(
            weight_df['CHARTTIME']
        ).dt.month.astype(str)
    ins['ref'] = pd.to_datetime(
            ins['GLCTIMER']
        ).dt.year.astype(str) + '-' + pd.to_datetime(
            ins['GLCTIMER']
        ).dt.month.astype(str)
    glc['ref'] = pd.to_datetime(
            glc['GLCTIMER']
        ).dt.year.astype(str) + '-' + pd.to_datetime(
            glc['GLCTIMER']
        ).dt.month.astype(str)

    ref_weight_df = weight_df.groupby([
        'SUBJECT_ID', 'HADM_ID', 'ref'
        ])['VALUE_kg'].agg(np.mean).reset_index()
    [ins, glc] = merge_dfs(
        [ins, glc],
        [ref_weight_df],
        keys=['SUBJECT_ID', 'HADM_ID', 'ref']
        )
    [ins, glc] = arrange_columns(
        [ins, glc],
        'fill',
        ['weight', 'VALUE_kg']
        )
    [ins, glc] = arrange_columns(
        [ins, glc],
        'drop',
        'VALUE_kg'
        )

    print(str(
        pd.isnull(ins['weight']).sum()
        ) + '\t weight data are still missing in treatment group')
    print(str(
        pd.isnull(glc['weight']).sum()
        ) + '\t weight data are still missing in controlled group')

    # attach remained weight based on year info
    weight_df['ref'] = pd.to_datetime(weight_df['CHARTTIME']).dt.year
    ins['ref'] = pd.to_datetime(ins['GLCTIMER']).dt.year
    glc['ref'] = pd.to_datetime(glc['GLCTIMER']).dt.year

    ref_weight_df = weight_df.groupby([
        'SUBJECT_ID', 'HADM_ID', 'ref'
        ])['VALUE_kg'].agg(np.mean).reset_index()
    [ins, glc] = merge_dfs(
        [ins, glc],
        [ref_weight_df],
        keys=['SUBJECT_ID', 'HADM_ID', 'ref']
        )
    [ins, glc] = arrange_columns(
        [ins, glc],
        'fill',
        ['weight', 'VALUE_kg']
        )
    [ins, glc] = arrange_columns(
        [ins, glc],
        'drop',
        'VALUE_kg'
        )

    print(str(
        pd.isnull(ins['weight']).sum()
        ) + '\t weight data are still missing in treatment group')
    print(str(
        pd.isnull(glc['weight']).sum()
        ) + '\t weight data are still missing in controlled group')

    ins.drop('ref', axis=1, inplace=True)
    glc.drop('ref', axis=1, inplace=True)

    if save:
        ins.to_csv(
            os.path.join(csv_path, 'glc_ins_wfeature.csv')
            , index=False)
        glc.to_csv(
            os.path.join(csv_path, 'glc_glc_wfeature.csv')
            , index=False)
    return ins, glc

try:
    data_path = sys.argv[1]
except IndexError:
    data_path = '../mimic-iii/mimic-iii-clinical-database-1.4/'

save_path = '../data/'
os.makedirs(save_path, exist_ok=True)

_, _, _ = extract_data(data_path)
_, _ = filter_data(data_path)
ins_df, glc_df = attach_data(data_path, save_path)
