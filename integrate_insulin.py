"""
RITE
Copyright 2023-present NAVER Cloud Corp.
MIT License
"""

import os
import sys
import pandas as pd
import numpy as np

try:
    data_path = sys.argv[1]
except IndexError:
    data_path = '../mimic-iii/mimic-iii-clinical-database-1.4/'
os.makedirs(os.path.join(data_path, 'processed'), exist_ok=True)

save_path = '../data/'
os.makediers(save_path, exist_ok=True)

## --------------------------------------------------------------------- ##
## 1. MV event
## --------------------------------------------------------------------- ##
insulin_mv = pd.read_csv(os.path.join(data_path, 'INPUTEVENTS_MV.csv'))

# select insulin events
insulin_mv = insulin_mv[insulin_mv['ITEMID'].isin([
                223257,
                223258,
                223259,
                223260,
                223261,
                223262,
                ])]

# save subject info
insulin_mv_subject = insulin_mv[[
    'SUBJECT_ID', 'HADM_ID'
    ]].drop_duplicates().reset_index()
insulin_mv_subject.to_csv(
    os.path.join(data_path, 'processed', 'insulin_mv_subject.csv')
    , index=False)

# remove invalid data
insulin_mv = insulin_mv[insulin_mv['STATUSDESCRIPTION'] != 'Rewritten']

# assign insulin type
insulin_mv['INSULINTYPE'] = np.nan
insulin_mv['INSULINTYPE'] = np.where(
    insulin_mv['ITEMID'] == 223257,
    'Intermediate',
    insulin_mv['INSULINTYPE']
    )
insulin_mv['INSULINTYPE'] = np.where(
    insulin_mv['ITEMID'] == 223258,
    'Short',
    insulin_mv['INSULINTYPE']
    )
insulin_mv['INSULINTYPE'] = np.where(
    insulin_mv['ITEMID'] == 223259,
    'Intermediate',
    insulin_mv['INSULINTYPE']
    )
insulin_mv['INSULINTYPE'] = np.where(
    insulin_mv['ITEMID'] == 223260,
    'Long',
    insulin_mv['INSULINTYPE']
    )
insulin_mv['INSULINTYPE'] = np.where(
    insulin_mv['ITEMID'] == 223261,
    'Intermediate',
    insulin_mv['INSULINTYPE']
    )
insulin_mv['INSULINTYPE'] = np.where(
    insulin_mv['ITEMID'] == 223262,
    'Short',
    insulin_mv['INSULINTYPE']
    )

# assign event type
insulin_mv['EVENT'] = np.nan
insulin_mv['EVENT'] = np.where(
    insulin_mv['ORDERCATEGORYNAME'].str.contains('NON IV', case=False, na=False),
    'BOLUS_INJECTION',
    insulin_mv['EVENT']
    )
insulin_mv['EVENT'] = np.where(
    insulin_mv['ORDERCATEGORYNAME'].str.contains('MED BOLUS', case=False, na=False),
    'BOLUS_PUSH',
    insulin_mv['EVENT']
    )
insulin_mv['EVENT'] = np.where(
    insulin_mv['ORDERCATEGORYNAME'].isin(['01-Drips', '12-Parenteral Nutrition']),
    'INFUSION',
    insulin_mv['EVENT']
    )

# mark infusion stoptime
insulin_mv['INFXSTOP'] = np.where(
    insulin_mv['STATUSDESCRIPTION'].isin(['Paused', 'Stopped']),
    1, 0
    )

# refine dataframe
insulin_mv[['ICUSTAY_ID', 'INFXSTOP', 'RATE', 'ORIGINALRATE']] = insulin_mv[
        ['ICUSTAY_ID', 'INFXSTOP', 'RATE', 'ORIGINALRATE']
        ].apply(pd.to_numeric, errors='coerce')
insulin_mv = insulin_mv[[
    'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'STARTTIME', 'ENDTIME', 'AMOUNT', 'RATE',
    'ORIGINALRATE', 'ITEMID', 'ORDERCATEGORYNAME', 'INSULINTYPE', 'EVENT', 'INFXSTOP'
    ]]
insulin_mv.dropna(subset=['ICUSTAY_ID'], inplace=True)

# filter infusions
MV_infusions = insulin_mv[
    (insulin_mv['INSULINTYPE']=="Short") &
    (insulin_mv['EVENT'].str.contains('INFUSION'))
    ].copy(deep=True)

# replace missing rates with original rate
MV_infusions['RATE'].fillna(MV_infusions['ORIGINALRATE'], inplace=True)

# extract values that are NOT null
MV_infusions = MV_infusions.dropna(subset=['RATE'])

# remove rates <= 0 U/hr
MV_infusions = MV_infusions[MV_infusions['RATE'] > 0]

# drop boluses over 99th percentile
MV_infusions = MV_infusions[
    (MV_infusions['RATE'] < (MV_infusions['RATE'].quantile(.99)))
    ]

# filter short-acting boluses
MV_bol_short = insulin_mv[
    (insulin_mv['INSULINTYPE']=="Short") &
    (insulin_mv['EVENT'].str.contains('BOLUS'))
    ].copy(deep=True)

# remove doses <= 0 U
MV_bol_short = MV_bol_short[MV_bol_short['AMOUNT'] > 0]

# drop boluses over 99th percentile
MV_bol_short = MV_bol_short[
    (MV_bol_short['AMOUNT'] < (MV_bol_short['AMOUNT'].quantile(.99)))
    ]

# filter intermediate-acting insulin
MV_bol_inter = insulin_mv[
    (insulin_mv['INSULINTYPE']=="Intermediate") &
    (insulin_mv['EVENT'].str.contains('BOLUS'))
    ].copy(deep=True)

# remove doses <= 0 U
MV_bol_inter = MV_bol_inter[MV_bol_inter['AMOUNT'] > 0]

# filter long-acting insulin boluses
MV_bol_long = insulin_mv[
    (insulin_mv['INSULINTYPE']=="Long") &
    (insulin_mv['EVENT'].str.contains('BOLUS'))
    ].copy(deep=True)

# remove doses <= 0 U
MV_bol_long = MV_bol_long[MV_bol_long['AMOUNT'] > 0]


frames = [MV_bol_short, MV_bol_inter, MV_infusions, MV_bol_long]
MV_insulin_step6 = pd.concat(
    frames,
    sort=True,
    verify_integrity=True,
    ignore_index=True,
    axis=0
    )
cols = list(MV_bol_short.columns)
MV_insulin_step6 = MV_insulin_step6[cols]

del frames, cols

insulin_cur = MV_insulin_step6[[
    'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'STARTTIME', 'ENDTIME', 'AMOUNT', 'RATE',
    'INSULINTYPE', 'EVENT', 'INFXSTOP'
    ]].copy(deep=True)
insulin_cur.rename(columns={
                    'AMOUNT': 'INPUT',
                    'RATE': 'INPUT_HRS'
                    }, inplace=True)
insulin_cur.to_csv(
    os.path.join(save_path, 'insulin.csv')
    , index=False
    )

## --------------------------------------------------------------------- ##
## 2. Glucose
## --------------------------------------------------------------------- ##
glucose = pd.read_csv(os.path.join(data_path, 'processed', 'glucose.csv'), index_col=0)

# refine glucose dataframe
glucose.rename(columns={
                'TIME': 'GLCTIMER',
                'VALUE': 'GLC',
                }, inplace=True)
glucose.drop_duplicates([
    'SUBJECT_ID', 'HADM_ID', 'GLCTIMER', 'GLC', 'GLCSOURCE'
    ], inplace=True)
glucose.sort_values(by=[
    'SUBJECT_ID', 'HADM_ID', 'GLCTIMER', 'GLCSOURCE', 'GLC'
    ], inplace=True)

# select the first value when multiple records exist
glucose = glucose.groupby([
    'SUBJECT_ID', 'HADM_ID', 'GLCTIMER'
    ]).agg('first').reset_index()

# load icu stay information
icu_stay = pd.read_csv(os.path.join(data_path, 'ICUSTAYS.csv'))

# prepare time information
icu_stay['INTIME'] = pd.to_datetime(icu_stay['INTIME'], utc=True)
icu_stay['OUTTIME'] = pd.to_datetime(icu_stay['OUTTIME'], utc=True)
glucose['GLCTIMER'] = pd.to_datetime(glucose['GLCTIMER'], utc=True)
insulin_cur['STARTTIME'] = pd.to_datetime(insulin_cur['STARTTIME'], utc=True)
insulin_cur['ENDTIME'] = pd.to_datetime(insulin_cur['ENDTIME'], utc=True)

# add admission information
icu_stay.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID'], inplace=True)
icu_stay['first_ICU_stay'] = np.where(
    ~icu_stay.duplicated(subset=['HADM_ID'], keep=False),
    1, 0
    )
icu_stay.loc[
    icu_stay.groupby(['SUBJECT_ID', 'HADM_ID', 'INTIME']).head(1).index, 'first_ICU_stay'
    ] = 1

glucose = glucose.merge(icu_stay, on=['SUBJECT_ID', 'HADM_ID'], how='left')
insulin_cur = insulin_cur.merge(
    icu_stay,
    on=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID'],
    how='left'
    )

# reject measurements occur before or after ICU stay
glucose = glucose[
    (glucose['GLCTIMER'] > glucose['INTIME']) &
    (glucose['GLCTIMER'] < glucose['OUTTIME'])
    ]
insulin_cur = insulin_cur[
    (insulin_cur['STARTTIME'] > insulin_cur['INTIME']) &
    (insulin_cur['ENDTIME'] < insulin_cur['OUTTIME'])
    ]

# reassign row ID
glucose = glucose.drop(
        columns=['ROW_ID']
    ).reset_index(drop=True).reset_index().rename(
        columns={'index': 'GLC_ROW_ID'}
    )
insulin_cur = insulin_cur.drop(
        columns=['ROW_ID']
    ).reset_index(drop=True).reset_index().rename(
        columns={'index': 'INSULIN_ROW_ID'}
    )

# integrate glucose and insulin
glc_ins = pd.concat((
    glucose[['GLC_ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'GLCTIMER', 'GLC']],
    insulin_cur[['INSULIN_ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'STARTTIME', 'ENDTIME']]
    ), axis=0)
glc_ins['TIMER'] = np.where(
    pd.isnull(glc_ins['GLCTIMER']),
    glc_ins['STARTTIME'],
    glc_ins['GLCTIMER']
    )
glc_ins.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'TIMER'], inplace=True)

## --------------------------------------------------------------------- ##
## 3. Pairing
## --------------------------------------------------------------------- ##
"""
Paring insuling and glucose measurements
 - Target insulin treatment (I*) was conducted based on reference glucose (ref-G) measurement.
 - For causal inference, fullowing glucose (fu-G) measurement is required.
 - Multiple insulin treatments between glucose measurements are regarded
    as a single multi-dimensional treatment.

 *G': a glucose measurement the time gap between which and insulin treatment is larger than 90 min
"""

base_keys = ['SUBJECT_ID', 'HADM_ID', 'GLC_ROW_ID', 'GLC', 'GLCTIMER']
add_keys = [
    'INSULIN_ROW_ID', 'INSULIN_TIMER', 'INSULIN_TIMER_GAP', 'FU_GLC_ROW_ID',
    'FU_GLC', 'FU_GLCTIMER', 'FU_GLCTIMER_GAP'
    ]
glc_ins_group = glc_ins.groupby(['SUBJECT_ID', 'HADM_ID'])

############# 1ST CLAUSE #############
# ref-G I* fu-G
glc_ins1 = glc_ins.copy()

# following insulin
glc_ins1['NEXT_TIMER'] = glc_ins_group['TIMER'].shift(-1)
glc_ins1['NEXT_INSULIN_TIMER_GAP'] = glc_ins1['NEXT_TIMER'] - glc_ins1['TIMER']
glc_ins1['NEXT_INSULIN_ROW_ID'] = glc_ins_group['INSULIN_ROW_ID'].shift(-1)

# following glucose
glc_ins1['NEXT_NEXT_TIMER'] = glc_ins_group['TIMER'].shift(-2)
glc_ins1['NEXT_NEXT_GLCTIMER_GAP'] = glc_ins1['NEXT_NEXT_TIMER'] - glc_ins1['TIMER']
glc_ins1['NEXT_NEXT_GLC_ROW_ID'] = glc_ins_group['GLC_ROW_ID'].shift(-2)
glc_ins1['NEXT_NEXT_GLC'] = glc_ins_group['GLC'].shift(-2)

cdt1 = (
        glc_ins1['NEXT_INSULIN_TIMER_GAP'] <= pd.Timedelta(90, 'm')
    ) & (
        glc_ins1['NEXT_INSULIN_TIMER_GAP'] <= (
            glc_ins1['NEXT_NEXT_GLCTIMER_GAP'] - glc_ins1['NEXT_INSULIN_TIMER_GAP']
    )) & (
        glc_ins1['GLC'] >= 90
    ) & (
        ~pd.isnull(glc_ins1['NEXT_INSULIN_ROW_ID'])
    ) & (
        glc_ins1['NEXT_NEXT_GLC'] < glc_ins1['GLC']
    ) & ((
        glc_ins1['NEXT_NEXT_GLCTIMER_GAP'] - glc_ins1['NEXT_INSULIN_TIMER_GAP']
        ) > pd.Timedelta(90, 'm')
    )

glc_ins1 = glc_ins1[cdt1]
add_key_targets = [
    'NEXT_INSULIN_ROW_ID', 'NEXT_TIMER', 'NEXT_INSULIN_TIMER_GAP',
    'NEXT_NEXT_GLC_ROW_ID', 'NEXT_NEXT_GLC', 'NEXT_NEXT_TIMER',
    'NEXT_NEXT_GLCTIMER_GAP'
    ]
glc_ins1 = glc_ins1[base_keys + add_key_targets]
glc_ins1.rename(columns=dict(zip(add_key_targets, add_keys)), inplace=True)

# attach glucose info
glc_ins1 = glc_ins1.merge(
    glucose[['GLC_ROW_ID', 'ITEMID', 'GLCSOURCE']].rename(
        columns={
            'ITEMID': 'GLC_ITEMID'
        }), on=['GLC_ROW_ID'], how='left')
glc_ins1 = glc_ins1.merge(
    glucose[['GLC_ROW_ID', 'ITEMID', 'GLCSOURCE']].rename(
        columns={
            'GLC_ROW_ID': 'FU_GLC_ROW_ID',
            'ITEMID': 'FU_GLC_ITEMID',
            'GLCSOURCE': 'FU_GLCSOURCE'
        }), on=['FU_GLC_ROW_ID'], how='left')

# attach insulin info
glc_ins1 = glc_ins1.merge(
    insulin_cur[[
        'INSULIN_ROW_ID', 'INPUT', 'INPUT_HRS', 'INSULINTYPE', 'EVENT', 'INFXSTOP'
        ]].rename(
            columns={
                'INPUT': 'INSULIN_INPUT',
                'INPUT_HRS': 'INSULIN_INPUT_HRS',
                'EVENT': 'INSULIN_EVENT',
                'INFXSTOP': 'INSULIN_INFXSTOP'
            }), on=['INSULIN_ROW_ID'], how='left')


############# 2ND CLAUSE #############
# G I* ref-G, fu-G
# when the charting of reference glucose was delayed
glc_ins2 = glc_ins.copy()

# preceding insulin
glc_ins2['PREV_TIMER'] = glc_ins_group['TIMER'].shift(1)
glc_ins2['PREV_INSULIN_TIMER_GAP'] = glc_ins2['TIMER'] - glc_ins2['PREV_TIMER']
glc_ins2['PREV_INSULIN_ROW_ID'] = glc_ins_group['INSULIN_ROW_ID'].shift(1)

# preceding glucose
glc_ins2['PREV_PREV_TIMER'] = glc_ins_group['TIMER'].shift(2)
glc_ins2['PREV_PREV_GLCTIMER_GAP'] = glc_ins2['TIMER'] - glc_ins2['PREV_PREV_TIMER']
glc_ins2['PREV_PREV_GLC_ROW_ID'] = glc_ins_group['GLC_ROW_ID'].shift(2)
glc_ins2['PREV_PREV_GLC'] = glc_ins_group['GLC'].shift(2)

# following glucose
glc_ins2['NEXT_TIMER'] = glc_ins_group['TIMER'].shift(-1)
glc_ins2['NEXT_GLCTIMER_GAP'] = glc_ins2['NEXT_TIMER'] - glc_ins2['TIMER']
glc_ins2['NEXT_GLC_ROW_ID'] = glc_ins_group['GLC_ROW_ID'].shift(-1)
glc_ins2['NEXT_GLC'] = glc_ins_group['GLC'].shift(-1)

cdt2 = ((
        glc_ins2['PREV_PREV_GLCTIMER_GAP'] - glc_ins2['PREV_INSULIN_TIMER_GAP']
        ) >= glc_ins2['PREV_INSULIN_TIMER_GAP']
    ) & ((
        glc_ins2['PREV_PREV_GLCTIMER_GAP'] - glc_ins2['PREV_INSULIN_TIMER_GAP']
        ) <= pd.Timedelta(90, 'm')
    ) & (
        glc_ins2['PREV_INSULIN_TIMER_GAP'] <= pd.Timedelta(90, 'm')
    ) & (
        glc_ins2['GLC'] >= 90
    ) & (
        ~pd.isnull(glc_ins2['PREV_INSULIN_ROW_ID'])
    ) & (
        glc_ins2['GLC'] > glc_ins2['PREV_PREV_GLC']
    ) & (
        glc_ins2['NEXT_GLC'] < glc_ins2['GLC']
    ) & ((
        glc_ins2['NEXT_GLCTIMER_GAP'] + glc_ins2['PREV_INSULIN_TIMER_GAP']
        ) > pd.Timedelta(90, 'm')
    )

glc_ins2 = glc_ins2[cdt2]
add_key_targets = [
    'PREV_INSULIN_ROW_ID', 'PREV_TIMER', 'PREV_INSULIN_TIMER_GAP', 'NEXT_GLC_ROW_ID',
    'NEXT_GLC', 'NEXT_TIMER', 'NEXT_GLCTIMER_GAP'
    ]
glc_ins2 = glc_ins2[base_keys + add_key_targets]
glc_ins2.rename(columns=dict(zip(add_key_targets, add_keys)), inplace=True)

# attach glucose info
glc_ins2 = glc_ins2.merge(
    glucose[['GLC_ROW_ID', 'ITEMID', 'GLCSOURCE']].rename(
        columns={
            'ITEMID': 'GLC_ITEMID'
        }), on=['GLC_ROW_ID'], how='left')
glc_ins2 = glc_ins2.merge(
    glucose[['GLC_ROW_ID', 'ITEMID', 'GLCSOURCE']].rename(
        columns={
            'GLC_ROW_ID': 'FU_GLC_ROW_ID',
            'ITEMID': 'FU_GLC_ITEMID',
            'GLCSOURCE': 'FU_GLCSOURCE'
        }), on=['FU_GLC_ROW_ID'], how='left')

# attach insulin info
glc_ins2 = glc_ins2.merge(
    insulin_cur[[
        'INSULIN_ROW_ID', 'INPUT', 'INPUT_HRS', 'INSULINTYPE', 'EVENT', 'INFXSTOP'
        ]].rename(
            columns={
                'INPUT': 'INSULIN_INPUT',
                'INPUT_HRS': 'INSULIN_INPUT_HRS',
                'EVENT': 'INSULIN_EVENT',
                'INFXSTOP': 'INSULIN_INFXSTOP'
                }), on=['INSULIN_ROW_ID'], how='left')

############# 3RD CLAUSE #############
# ref-G I I* fu-G OR ref-G I* I fu-G
# when two insulin events occured between glucose measurement
glc_ins3 = glc_ins.copy()

# following 1st insulin
glc_ins3['NEXT_TIMER'] = glc_ins_group['TIMER'].shift(-1)
glc_ins3['NEXT_INSULIN_TIMER_GAP'] = glc_ins3['NEXT_TIMER'] - glc_ins3['TIMER']
glc_ins3['NEXT_INSULIN_ROW_ID'] = glc_ins_group['INSULIN_ROW_ID'].shift(-1)

# following 2nd insulin
glc_ins3['NEXT_NEXT_TIMER'] = glc_ins_group['TIMER'].shift(-2)
glc_ins3['NEXT_NEXT_INSULIN_TIMER_GAP'] = glc_ins3['NEXT_NEXT_TIMER'] - glc_ins3['TIMER']
glc_ins3['NEXT_NEXT_INSULIN_ROW_ID'] = glc_ins_group['INSULIN_ROW_ID'].shift(-2)

# following glucose
glc_ins3['NEXT_NEXT_NEXT_TIMER'] = glc_ins_group['TIMER'].shift(-3)
glc_ins3['NEXT_NEXT_NEXT_GLCTIMER_GAP'] = glc_ins3['NEXT_NEXT_NEXT_TIMER'] - glc_ins3['TIMER']
glc_ins3['NEXT_NEXT_NEXT_GLC_ROW_ID'] = glc_ins_group['GLC_ROW_ID'].shift(-3)
glc_ins3['NEXT_NEXT_NEXT_GLC'] = glc_ins_group['GLC'].shift(-3)

cdt3 = (
    glc_ins3['NEXT_NEXT_INSULIN_TIMER_GAP'] <= pd.Timedelta(90, 'm')
    ) & (
        glc_ins3['NEXT_NEXT_INSULIN_TIMER_GAP'] <= (
            glc_ins3['NEXT_NEXT_NEXT_GLCTIMER_GAP'] - glc_ins3['NEXT_NEXT_INSULIN_TIMER_GAP']
    )) & (
        glc_ins3['GLC'] >= 90
    ) & (
        ~pd.isnull(glc_ins3['NEXT_INSULIN_ROW_ID'])
    ) & (
        ~pd.isnull(glc_ins3['NEXT_NEXT_INSULIN_ROW_ID'])
    ) & (
        glc_ins3['GLC'] > glc_ins3['NEXT_NEXT_NEXT_GLC']
    ) & ((
        glc_ins3['NEXT_NEXT_NEXT_GLCTIMER_GAP'] - glc_ins3['NEXT_NEXT_INSULIN_TIMER_GAP']
        ) > pd.Timedelta(90, 'm')
    )

glc_ins3 = glc_ins3[cdt3]

# attach glucose info
glc_ins3 = glc_ins3.merge(
    glucose[['GLC_ROW_ID', 'ITEMID', 'GLCSOURCE']].rename(
        columns={
            'ITEMID': 'GLC_ITEMID'
        }), on=['GLC_ROW_ID'], how='left')
glc_ins3 = glc_ins3.merge(
    glucose[['GLC_ROW_ID', 'ITEMID', 'GLCSOURCE']].rename(
        columns={
            'GLC_ROW_ID': 'NEXT_NEXT_NEXT_GLC_ROW_ID',
            'ITEMID': 'FU_GLC_ITEMID',
            'GLCSOURCE': 'FU_GLCSOURCE'
        }), on=['NEXT_NEXT_NEXT_GLC_ROW_ID'], how='left')

# attach insulin info
glc_ins3 = glc_ins3.merge(
    insulin_cur[[
        'INSULIN_ROW_ID', 'INPUT', 'INPUT_HRS', 'INSULINTYPE', 'EVENT', 'INFXSTOP'
        ]].rename(
            columns={
                'INSULIN_ROW_ID': 'NEXT_INSULIN_ROW_ID',
                'INPUT': 'NEXT_INSULIN_INPUT',
                'INPUT_HRS': 'NEXT_INSULIN_INPUT_HRS',
                'INSULINTYPE': 'NEXT_INSULINTYPE',
                'EVENT': 'NEXT_INSULIN_EVENT',
                'INFXSTOP': 'NEXT_INSULIN_INFXSTOP'
            }), on=['NEXT_INSULIN_ROW_ID'], how='left')
glc_ins3 = glc_ins3.merge(
    insulin_cur[[
        'INSULIN_ROW_ID', 'INPUT', 'INPUT_HRS', 'INSULINTYPE', 'EVENT', 'INFXSTOP'
        ]].rename(
            columns={
                'INSULIN_ROW_ID': 'NEXT_NEXT_INSULIN_ROW_ID',
                'INPUT': 'NEXT_NEXT_INSULIN_INPUT',
                'INPUT_HRS': 'NEXT_NEXT_INSULIN_INPUT_HRS',
                'INSULINTYPE': 'NEXT_NEXT_INSULINTYPE',
                'EVENT': 'NEXT_NEXT_INSULIN_EVENT',
                'INFXSTOP': 'NEXT_NEXT_INSULIN_INFXSTOP'
            }), on=['NEXT_NEXT_INSULIN_ROW_ID'], how='left')

# insulin integration
glc_ins3['INSULIN_TIMER'] = glc_ins3.apply(
    lambda x: list([
        x['NEXT_TIMER'],
        x['NEXT_NEXT_TIMER']
        ]), axis=1)
glc_ins3['INSULIN_TIMER_GAP'] = glc_ins3.apply(
    lambda x: list([
        x['NEXT_INSULIN_TIMER_GAP'],
        x['NEXT_NEXT_INSULIN_TIMER_GAP']
        ]), axis=1)
glc_ins3['INSULIN_ROW_ID'] = glc_ins3.apply(
    lambda x: list([
        x['NEXT_INSULIN_ROW_ID'],
        x['NEXT_NEXT_INSULIN_ROW_ID']
        ]), axis=1)
glc_ins3['INSULIN_INPUT'] = glc_ins3.apply(
    lambda x: list([
        x['NEXT_INSULIN_INPUT'],
        x['NEXT_NEXT_INSULIN_INPUT']
        ]), axis=1)
glc_ins3['INSULIN_INPUT_HRS'] = glc_ins3.apply(
    lambda x: list([
        x['NEXT_INSULIN_INPUT_HRS'],
        x['NEXT_NEXT_INSULIN_INPUT_HRS']
        ]), axis=1)
glc_ins3['INSULINTYPE'] = glc_ins3.apply(
    lambda x: list([
        x['NEXT_INSULINTYPE'],
        x['NEXT_NEXT_INSULINTYPE']
        ]), axis=1)
glc_ins3['INSULIN_EVENT'] = glc_ins3.apply(
    lambda x: list([
        x['NEXT_INSULIN_EVENT'],
        x['NEXT_NEXT_INSULIN_EVENT']
        ]), axis=1)
glc_ins3['INSULIN_INFXSTOP'] = glc_ins3.apply(
    lambda x: list([
        x['NEXT_INSULIN_INFXSTOP'],
        x['NEXT_NEXT_INSULIN_INFXSTOP']
        ]), axis=1)

add_key_targets = [
    'INSULIN_ROW_ID', 'INSULIN_TIMER', 'INSULIN_TIMER_GAP',
    'NEXT_NEXT_NEXT_GLC_ROW_ID', 'NEXT_NEXT_NEXT_GLC', 'NEXT_NEXT_NEXT_TIMER',
    'NEXT_NEXT_NEXT_GLCTIMER_GAP'
    ]
glc_ins3 = glc_ins3[base_keys + add_key_targets + [
    'GLC_ITEMID', 'GLCSOURCE', 'FU_GLC_ITEMID', 'FU_GLCSOURCE', 'INSULIN_INPUT',
    'INSULIN_INPUT_HRS', 'INSULINTYPE', 'INSULIN_EVENT', 'INSULIN_INFXSTOP'
    ]]
glc_ins3.rename(columns=dict(zip(add_key_targets, add_keys)), inplace=True)

############# 4TH CLAUSE #############
# G I I* ref-G fu-G
# 3rd clause + when the glucose was lately recorded
glc_ins4 = glc_ins.copy()

# preceding 1st insulin
glc_ins4['PREV_TIMER'] = glc_ins_group['TIMER'].shift(1)
glc_ins4['PREV_INSULIN_TIMER_GAP'] = glc_ins4['TIMER'] - glc_ins4['PREV_TIMER']
glc_ins4['PREV_INSULIN_ROW_ID'] = glc_ins_group['INSULIN_ROW_ID'].shift(1)

# preceding 2nd insulin
glc_ins4['PREV_PREV_TIMER'] = glc_ins_group['TIMER'].shift(2)
glc_ins4['PREV_PREV_INSULIN_TIMER_GAP'] = glc_ins4['TIMER'] - glc_ins4['PREV_PREV_TIMER']
glc_ins4['PREV_PREV_INSULIN_ROW_ID'] = glc_ins_group['INSULIN_ROW_ID'].shift(2)

# preceding glucose
glc_ins4['PREV_PREV_PREV_TIMER'] = glc_ins_group['TIMER'].shift(3)
glc_ins4['PREV_PREV_PREV_GLCTIMER_GAP'] = glc_ins4['TIMER'] - glc_ins4['PREV_PREV_PREV_TIMER']
glc_ins4['PREV_PREV_PREV_GLC_ROW_ID'] = glc_ins_group['GLC_ROW_ID'].shift(3)
glc_ins4['PREV_PREV_PREV_GLC'] = glc_ins_group['GLC'].shift(3)

# following glucose
glc_ins4['NEXT_TIMER'] = glc_ins_group['TIMER'].shift(-1)
glc_ins4['NEXT_GLCTIMER_GAP'] = glc_ins4['NEXT_TIMER'] - glc_ins4['TIMER']
glc_ins4['NEXT_GLC_ROW_ID'] = glc_ins_group['GLC_ROW_ID'].shift(-1)
glc_ins4['NEXT_GLC'] = glc_ins_group['GLC'].shift(-1)

cdt4 = (
    glc_ins4['GLC'] >= 90
    ) & (
        ~pd.isnull(glc_ins4['PREV_INSULIN_ROW_ID'])
    ) & (
        ~pd.isnull(glc_ins4['PREV_PREV_INSULIN_ROW_ID'])
    ) & (
        glc_ins4['NEXT_GLC'] < glc_ins4['GLC']
    ) & ((
        glc_ins4['PREV_INSULIN_TIMER_GAP'] + glc_ins4['NEXT_GLCTIMER_GAP']
        ) > pd.Timedelta(90, 'm')
    ) & (
        glc_ins4['PREV_PREV_INSULIN_TIMER_GAP'] <= pd.Timedelta(90, 'm')
    ) & (
        glc_ins4['PREV_PREV_PREV_GLC'] < glc_ins4['GLC']
    ) & ((
        glc_ins4['PREV_PREV_PREV_GLCTIMER_GAP'] - glc_ins4['PREV_PREV_INSULIN_TIMER_GAP']
        ) > glc_ins4['PREV_PREV_INSULIN_TIMER_GAP']
    ) & ((
        glc_ins4['PREV_PREV_PREV_GLCTIMER_GAP'] - glc_ins4['PREV_INSULIN_TIMER_GAP']
        ) <= pd.Timedelta(90, 'm')
    )

glc_ins4 = glc_ins4[cdt4]

# attach glucose info
glc_ins4 = glc_ins4.merge(
    glucose[['GLC_ROW_ID', 'ITEMID', 'GLCSOURCE']].rename(
        columns={
            'ITEMID': 'GLC_ITEMID'
        }), on=['GLC_ROW_ID'], how='left')
glc_ins4 = glc_ins4.merge(
    glucose[['GLC_ROW_ID', 'ITEMID', 'GLCSOURCE']].rename(
        columns={
            'GLC_ROW_ID': 'NEXT_GLC_ROW_ID',
            'ITEMID': 'FU_GLC_ITEMID',
            'GLCSOURCE': 'FU_GLCSOURCE'
        }), on=['NEXT_GLC_ROW_ID'], how='left')

# attach insulin info
glc_ins4 = glc_ins4.merge(
    insulin_cur[[
        'INSULIN_ROW_ID', 'INPUT', 'INPUT_HRS', 'INSULINTYPE', 'EVENT', 'INFXSTOP'
        ]].rename(
            columns={
                'INSULIN_ROW_ID': 'PREV_INSULIN_ROW_ID',
                'INPUT': 'PREV_INSULIN_INPUT',
                'INPUT_HRS': 'PREV_INSULIN_INPUT_HRS',
                'INSULINTYPE': 'PREV_INSULINTYPE',
                'EVENT': 'PREV_INSULIN_EVENT',
                'INFXSTOP': 'PREV_INSULIN_INFXSTOP'
            }), on=['PREV_INSULIN_ROW_ID'], how='left')
glc_ins4 = glc_ins4.merge(
    insulin_cur[[
        'INSULIN_ROW_ID', 'INPUT', 'INPUT_HRS', 'INSULINTYPE', 'EVENT', 'INFXSTOP'
        ]].rename(
            columns={
                'INSULIN_ROW_ID': 'PREV_PREV_INSULIN_ROW_ID',
                'INPUT': 'PREV_PREV_INSULIN_INPUT',
                'INPUT_HRS': 'PREV_PREV_INSULIN_INPUT_HRS',
                'INSULINTYPE': 'PREV_PREV_INSULINTYPE',
                'EVENT': 'PREV_PREV_INSULIN_EVENT',
                'INFXSTOP': 'PREV_PREV_INSULIN_INFXSTOP'
            }), on=['PREV_PREV_INSULIN_ROW_ID'], how='left')

# insulin integration
glc_ins4['INSULIN_TIMER'] = glc_ins4.apply(
    lambda x: list([
        x['PREV_PREV_TIMER'],
        x['PREV_TIMER']
        ]), axis=1)
glc_ins4['INSULIN_TIMER_GAP'] = glc_ins4.apply(
    lambda x: list([
        x['PREV_PREV_INSULIN_TIMER_GAP'],
        x['PREV_INSULIN_TIMER_GAP']
        ]), axis=1)
glc_ins4['INSULIN_ROW_ID'] = glc_ins4.apply(
    lambda x: list([
        x['PREV_PREV_INSULIN_ROW_ID'],
        x['PREV_INSULIN_ROW_ID']
        ]), axis=1)
glc_ins4['INSULIN_INPUT'] = glc_ins4.apply(
    lambda x: list([
        x['PREV_PREV_INSULIN_INPUT'],
        x['PREV_INSULIN_INPUT']
        ]), axis=1)
glc_ins4['INSULIN_INPUT_HRS'] = glc_ins4.apply(
    lambda x: list([
        x['PREV_PREV_INSULIN_INPUT_HRS'],
        x['PREV_INSULIN_INPUT_HRS']
        ]), axis=1)
glc_ins4['INSULINTYPE'] = glc_ins4.apply(
    lambda x: list([
        x['PREV_PREV_INSULINTYPE'],
        x['PREV_INSULINTYPE']
        ]), axis=1)
glc_ins4['INSULIN_EVENT'] = glc_ins4.apply(
    lambda x: list([
        x['PREV_PREV_INSULIN_EVENT'],
        x['PREV_INSULIN_EVENT']
        ]), axis=1)
glc_ins4['INSULIN_INFXSTOP'] = glc_ins4.apply(
    lambda x: list([
        x['PREV_PREV_INSULIN_INFXSTOP'],
        x['PREV_INSULIN_INFXSTOP']
        ]), axis=1)

add_key_targets = [
    'INSULIN_ROW_ID', 'INSULIN_TIMER', 'INSULIN_TIMER_GAP', 'NEXT_GLC_ROW_ID',
    'NEXT_GLC', 'NEXT_TIMER', 'NEXT_GLCTIMER_GAP'
    ]
glc_ins4 = glc_ins4[base_keys + add_key_targets + [
    'GLC_ITEMID', 'GLCSOURCE', 'FU_GLC_ITEMID', 'FU_GLCSOURCE', 'INSULIN_INPUT',
    'INSULIN_INPUT_HRS', 'INSULINTYPE', 'INSULIN_EVENT', 'INSULIN_INFXSTOP'
    ]]
glc_ins4.rename(columns=dict(zip(add_key_targets, add_keys)), inplace=True)

## --------------------------------------------------------------------- ##
## 4. Integration
## --------------------------------------------------------------------- ##
# save treatment group data
glc_ins_total = pd.concat((glc_ins1, glc_ins2, glc_ins3, glc_ins4), axis=0)
glc_ins_total.to_csv(
    os.path.join(save_path, 'glc_ins.csv')
    , index=False
    )

## --------------------------------------------------------------------- ##
## 5. Treatment
## --------------------------------------------------------------------- ##
# ref-G fu-G
# both glucose measurement are not matched with an insulin treatment
# no insulin before and after 90 minitues of glucose measurements

# follow-up glucose
glc_ins['NEXT_TIMER'] = glc_ins_group['TIMER'].shift(-1)
glc_ins['NEXT_GLCTIMER_GAP'] = glc_ins['NEXT_TIMER'] - glc_ins['TIMER']
glc_ins['NEXT_GLC_ROW_ID'] = glc_ins_group['GLC_ROW_ID'].shift(-1)
glc_ins['NEXT_GLC'] = glc_ins_group['GLC'].shift(-1)

# define valid range
glc_ins['EARLIEST'] = glc_ins['TIMER'] - pd.Timedelta(90, 'm')
glc_ins['LATEST'] = glc_ins['TIMER'] + pd.Timedelta(90, 'm')
glc = glc_ins[glc_ins['GLC'] >=90]

# check insulin treatment
ins_cdt = []
for _, ref_glc in glc.iterrows():
    current_target = glc_ins[(
            glc_ins['SUBJECT_ID'] == ref_glc['SUBJECT_ID']
        ) & (
            glc_ins['HADM_ID'] == ref_glc['HADM_ID']
        )]
    current_target = current_target[current_target['TIMER'].between(
        ref_glc['EARLIEST'], ref_glc['LATEST']
        )]
    if pd.isnull(current_target['INSULIN_ROW_ID']).min() > 0:
        ins_cdt.append(True)
    else:
        ins_cdt.append(False)

# make control group
glc_glc = glc[ins_cdt]
glc_glc.rename(columns={
        'NEXT_TIMER': 'FU_GLCTIMER',
        'NEXT_GLCTIMER_GAP': 'FU_GLCTIMER_GAP',
        'NEXT_GLC_ROW_ID': 'FU_GLC_ROW_ID',
        'NEXT_GLC': 'FU_GLC'
        }, inplace=True)

# attach necessary information
glc_glc = glc_glc.merge(
    glucose[['GLC_ROW_ID', 'ITEMID', 'GLCSOURCE']].rename(
        columns={
            'ITEMID': 'GLC_ITEMID'
        }), on=['GLC_ROW_ID'], how='left')
glc_glc = glc_glc.merge(
    glucose[['GLC_ROW_ID', 'ITEMID', 'GLCSOURCE']].rename(
        columns={
            'GLC_ROW_ID': 'FU_GLC_ROW_ID',
            'ITEMID': 'FU_GLC_ITEMID',
            'GLCSOURCE': 'FU_GLCSOURCE'
        }), on=['FU_GLC_ROW_ID'], how='left')

# save control group data
glc_glc = glc_glc[[
    'SUBJECT_ID', 'HADM_ID', 'GLC_ROW_ID', 'GLC', 'GLCTIMER', 'INSULIN_ROW_ID',
    'FU_GLC_ROW_ID', 'FU_GLC', 'FU_GLCTIMER', 'FU_GLCTIMER_GAP', 'GLC_ITEMID',
    'GLCSOURCE', 'FU_GLC_ITEMID', 'FU_GLCSOURCE'
    ]]
glc_glc.to_csv(
    os.path.join(save_path, 'glc_glc.csv')
    , index=False
    )
