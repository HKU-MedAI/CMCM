from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import re
import pandas as pd
from pandas import DataFrame, Series

from mimic3benchmark.util import dataframe_from_csv

###############################
# Non-time series preprocessing
###############################

g_map = {'F': 1, 'M': 2, 'OTHER': 3, '': 0}


def transform_gender(gender_series):
    global g_map
    return {'Gender': gender_series.fillna('').apply(lambda s: g_map[s] if s in g_map else g_map['OTHER'])}


e_map = {'ASIAN': 1,
         'BLACK': 2,
         'CARIBBEAN ISLAND': 2,
         'HISPANIC': 3,
         'SOUTH AMERICAN': 3,
         'WHITE': 4,
         'MIDDLE EASTERN': 4,
         'PORTUGUESE': 4,
         'AMERICAN INDIAN': 0,
         'NATIVE HAWAIIAN': 0,
         'UNABLE TO OBTAIN': 0,
         'PATIENT DECLINED TO ANSWER': 0,
         'UNKNOWN': 0,
         'OTHER': 0,
         '': 0}


def transform_ethnicity(ethnicity_series):
    global e_map

    def aggregate_ethnicity(ethnicity_str):
        return ethnicity_str.replace(' OR ', '/').split(' - ')[0].split('/')[0]

    ethnicity_series = ethnicity_series.apply(aggregate_ethnicity)
    return {'Ethnicity': ethnicity_series.fillna('').apply(lambda s: e_map[s] if s in e_map else e_map['OTHER'])}


def assemble_episodic_data(stays, diagnoses):
    data = {'Icustay': stays.stay_id, 'Age': stays.age, 'Length of Stay': stays.los,
            'Mortality': stays.mortality}
    data.update(transform_gender(stays.gender))
    data.update(transform_ethnicity(stays.race))
    data['Height'] = np.nan
    data['Weight'] = np.nan
    data = DataFrame(data).set_index('Icustay')
    data = data[['Ethnicity', 'Gender', 'Age', 'Height', 'Weight', 'Length of Stay', 'Mortality']]
    return data.merge(extract_diagnosis_labels(diagnoses), left_index=True, right_index=True)


# diagnosis_labels = ['4019', '4280', '41401', '42731', '25000', '5849', '2724', '51881', '53081', '5990', '2720',
#                     '2859', '2449', '486', '2762', '2851', '496', 'V5861', '99592', '311', '0389', '5859', '5070',
#                     '40390', '3051', '412', 'V4581', '2761', '41071', '2875', '4240', 'V1582', 'V4582', 'V5867',
#                     '4241', '40391', '78552', '5119', '42789', '32723', '49390', '9971', '2767', '2760', '2749',
#                     '4168', '5180', '45829', '4589', '73300', '5845', '78039', '5856', '4271', '4254', '4111',
#                     'V1251', '30000', '3572', '60000', '27800', '41400', '2768', '4439', '27651', 'V4501', '27652',
#                     '99811', '431', '28521', '2930', '7907', 'E8798', '5789', '79902', 'V4986', 'V103', '42832',
#                     'E8788', '00845', '5715', '99591', '07054', '42833', '4275', '49121', 'V1046', '2948', '70703',
#                     '2809', '5712', '27801', '42732', '99812', '4139', '3004', '2639', '42822', '25060', 'V1254',
#                     '42823', '28529', 'E8782', '30500', '78791', '78551', 'E8889', '78820', '34590', '2800', '99859',
#                     'V667', 'E8497', '79092', '5723', '3485', '5601', '25040', '570', '71590', '2869', '2763', '5770',
#                     'V5865', '99662', '28860', '36201', '56210']


diagnosis_labels = ['I169', 'I509', 'I2510', 'I4891', 'E119', 'N179', 'E785', 'J9690',
'K219', 'N390', 'E7801', 'D649', 'E039', 'J189', 'E872', 'D62', 'J449', 'Z7901', 'R6520',
'F329', 'A419', 'N189', 'J690', 'I129', 'F17200', 'I252', 'Z951', 'E871', 'I214', 'D696',
'I348', 'Z87891', 'Z9861', 'Z794', 'I359', 'I120', 'R6521', 'J918', 'R001', 'G4733', 'J45998',
'I9789', 'E875', 'E870', 'M109', 'I2789', 'J9819', 'I9581', 'I959', 'M810', 'N170', 'R569', 'N186',
'I472', 'I428', 'I200', 'Z86718', 'F419', 'E1342', 'N400', 'E669', 'I2510', 'E876', 'I739', 'E860',
'Z950', 'E861', 'N99821', 'I619', 'D631', 'F05', 'R7881', 'Y848', 'K922', 'R0902', 'Z66', 'Z853',
'I5032', 'Y838', 'A0472', 'K7469', 'A419', 'B182', 'I5033', 'I469', 'J441', 'Z8546', 'F068',
'L89159', 'D509', 'K7030', 'E6601', 'I4892', 'N99841', 'I209', 'F341', 'E46', 'I5022', 'E1140',
'Z8673', 'I5023', 'D638', 'Y832', 'F1010', 'R197', 'R570', 'W19XXXA', 'R339', 'G40909', 'D500',
'T814XXA', 'Z515', 'Y92199', 'R791', 'K766', 'G936', 'K567', 'E1129', 'K762', 'M1990', 'D689',
'E873', 'K8592', 'Z7952', 'T827XXA', 'D72829', 'E11319', 'K5730', '4019', '4280', '41401',
'42731', '25000', '5849', '2724', '51881', '53081', '5990', '2720', '2859', '2449', '486',
'2762', '2851', '496', 'V5861', '99592', '311', '0389', '5859', '5070', '40390', '3051',
'412', 'V4581', '2761', '41071', '2875', '4240', 'V1582', 'V4582', 'V5867', '4241', '40391',
'78552', '5119', '42789', '32723', '49390', '9971', '2767', '2760', '2749', '4168', '5180',
'45829', '4589', '73300', '5845', '78039', '5856', '4271', '4254', '4111', 'V1251', '30000',
'3572', '60000', '27800', '41400', '2768', '4439', '27651', 'V4501', '27652', '99811', '431',
'28521', '2930', '7907', 'E8798', '5789', '79902', 'V4986', 'V103', '42832', 'E8788', '00845',
'5715', '99591', '07054', '42833', '4275', '49121', 'V1046', '2948', '70703', '2809', '5712',
'27801', '42732', '99812', '4139', '3004', '2639', '42822', '25060', 'V1254', '42823', '28529',
'E8782', '30500', '78791', '78551', 'E8889', '78820', '34590', '2800', '99859', 'V667', 'E8497',
'79092', '5723', '3485', '5601', '25040', '570', '71590', '2869', '2763', '5770', 'V5865', '99662',
'28860', '36201', '56210']
def extract_diagnosis_labels(diagnoses):
    global diagnosis_labels
    diagnoses['value'] = 1
    labels = diagnoses[['stay_id', 'icd_code', 'value']].drop_duplicates()\
                      .pivot(index='stay_id', columns='icd_code', values='value').fillna(0).astype(int)
    for l in diagnosis_labels:
        if l not in labels:
            labels[l] = 0
    labels = labels[diagnosis_labels]
    return labels.rename(dict(zip(diagnosis_labels, ['Diagnosis ' + d for d in diagnosis_labels])), axis=1)


def add_hcup_ccs_2015_groups(diagnoses, definitions):

    def_map = {}
    for dx in definitions:
        for code in definitions[dx]['codes']:
            def_map[code] = (dx, definitions[dx]['use_in_benchmark'])
    
    diagnoses['HCUP_CCS_2015'] = diagnoses.icd_code.apply(lambda c: def_map[c][0] if c in def_map else None)
    diagnoses['USE_IN_BENCHMARK'] = diagnoses.icd_code.apply(lambda c: int(def_map[c][1]) if c in def_map else None)
    
    return diagnoses


def make_phenotype_label_matrix(phenotypes, stays=None):

    phenotypes = phenotypes[['stay_id', 'HCUP_CCS_2015']].loc[phenotypes.USE_IN_BENCHMARK > 0].drop_duplicates()
    phenotypes['value'] = 1
    phenotypes = phenotypes.pivot(index='stay_id', columns='HCUP_CCS_2015', values='value')
    if stays is not None:
        phenotypes = phenotypes.reindex(stays.stay_id.sort_values())
    
    return phenotypes.fillna(0).astype(int).sort_index(axis=0).sort_index(axis=1)
    # import pdb; pdb.set_trace()


###################################
# Time series preprocessing
###################################

def read_itemid_to_variable_map(fn, variable_column='LEVEL2'):

    var_map = pd.read_csv(fn).fillna('').astype(str)
    # var_map[variable_column] = var_map[variable_column].apply(lambda s: s.lower())
    var_map.COUNT = var_map.COUNT.astype(int)
    var_map = var_map[(var_map[variable_column] != '') & (var_map.COUNT > 0)]
    var_map = var_map[(var_map.STATUS == 'ready')]
    var_map.ITEMID = var_map.ITEMID.astype(int)
    var_map = var_map[[variable_column, 'ITEMID', 'MIMIC LABEL']]
    # .set_index('ITEMID')
    var_map = var_map.rename({variable_column: 'variable', 'MIMIC LABEL': 'mimic_label'}, axis=1)
    # var_map.co
    var_map.columns = var_map.columns.str.lower()
    # import pdb; pdb.set_trace()

    return var_map


def map_itemids_to_variables(events, var_map):
    # import pdb; pdb.set_trace()
    # v_a = var_map.itemid.values
    # v_b = events.itemid.values
    # np.intersect1d(v_a, v_b)
    return events.merge(var_map, left_on='itemid', right_on='itemid') #right_index=True)


def read_variable_ranges(fn, variable_column='LEVEL2'):
    columns = [variable_column, 'OUTLIER LOW', 'VALID LOW', 'IMPUTE', 'VALID HIGH', 'OUTLIER HIGH']
    to_rename = dict(zip