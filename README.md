## Real-world data for Insulin Treatment Effect (RITE)

Python codes to build RITE dataset from [MIMIC-III Clinical Database](https://physionet.org/content/mimiciii/1.4/) | Paper (TBU)

**Seong-Eun Moon, Yera Choi, Sewon Kim.**

Clova AI Lab Healthcare AI, NAVER Corp.

Our implementation is partially based on [this project](https://physionet.org/content/glucose-management-mimic/1.0.1/)

## Getting Started

### Requirements
- Python3
- NumPy
- pandas
- joblib
### Attributes
Attribute|Description
:--:|:--
SUBJECT_ID|unique ID of patients
HADM_ID|hospital admission ID
GLC_ROW_ID|row index of glucose measurement in the original glucose data
GLC|measured glucose values
GLCTIMER|timestamp for glucose measurements
INSULIN_ROW_ID|row index of insulin information in the origianl insulin data
INSULIN_TIMER|timestamp for insulin treatment
INSULIN_TIMER_GAP|time gap between the reference glucose measurement and insulin treatment
FU_GLC_ROW_ID|row index for follow-up glucose measurement (after insulin treatment)
FU_GLC|glucose value of follow-up measurement
FU_GLCTIMER|timestamp for follow-up glucose measurement
FU_GLCTIMER_GAP|time gap between the reference and follow-up glucose measurements
GLC_ITEMID|item ID of glucose measurement (including the information of measurement method)
GLCSOURCE|source of glucose measurement (e.g., finger stick)
FU_GLC_ITEMID|item ID of follow-up glucose measurement
FU_GLCSOURCE|source of follow-up glucose measurement
INSULIN_INPUT|amount of insulin
INSULIN_INPUT_HRS|consumed hours for insulin treatment specified when the insulin is intravenous infusion type
INSULINTYPE|acting type of insulin, i.e., the influence of insulin appears after a {short, intermediate, or long} interval
INSULIN_EVENT|type of insulin treatment
INSULIN_INFXSTOP|whether the insulin infusion was stopped
HEIGHT|height of patients
ADMISSION_TYPE|'elective', 'urgent', 'newborn', or 'emergency'
ETHNICITY|ethnicity of patients
DIAGNOSIS|Diagnosis in plain text
GENDER|gender of patients
AGE|patients' age at admission
weight|weight of patients
### Data preparation
1. Register for [PhysioNet](https://physionet.org/)
2. Credentialize your PhysioNet account to acquire access to MIMIC database (instructions can be found from [here](https://mimic.mit.edu/docs/gettingstarted/))
3. Download MIMIC-III
### RITE dataset generation
1. For cross-sectional data
```bash
python extract_glucose.py [path to MIMIC-III] && \
python integrate_insulin.py [path to MIMIC-III] && \
python extract_add_info.py [path to MIMIC-III]
```
2. Additionally, for longitudinal data
```bash
python make_longitudinal_data.py [path to MIMIC-III] && \
python generate_data.py
```
3. Dataset can be found in '../data' 

## License
```
Copyright 2022-present NAVER Cloud Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Acknowledgement
[extract_glucose.py](extract_glucose.py) and [integrate_insulin.py](integrate_insulin.py) are partially based on [Curated Data for Describing Blood Glucose Management in the Intensive Care Unit](https://physionet.org/content/glucose-management-mimic/1.0.1/), under [PhysioNet Credentialed Health Data License 1.5.0](https://physionet.org/content/glucose-management-mimic/view-license/1.0.1/).
