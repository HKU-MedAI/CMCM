import pandas as pd
import numpy as np

ehr_data_dir = "MedFuse/data/in-hospital-mortality"
cxr_data_resize_dir = "MedFuse/data/mimic-cxr"
cxr_splits = pd.read_csv(f"{cxr_data_resize_dir}/mimic-cxr-2.0.0-split.csv")
cxr_note_section = pd.read_csv(f"{cxr_data_resize_dir}/mimic_cxr_sectioned.csv")
cxr_note_section["study_id"] = cxr_note_section["study"].str[1:].astype(int)

print(f"before update, cxr image No: {cxr_splits.split.value_counts()}")
print(f"before update, cxr note No: {len(cxr_note_section)}")

cxr_splits = cxr_splits.merge(cxr_note_section, how="inner", on="study_id")
print(f"after merge note, cxr image No: {cxr_splits.split.value_counts()}")

ehr_train = pd.read_csv(f"{ehr_data_dir}/train_listfile.csv")
ehr_split_val = pd.read_csv(f"{ehr_data_dir}/val_listfile.csv")
ehr_split_test = pd.read_csv(f"{ehr_data_dir}/test_listfile.csv")

train_subject_ids = [int(stay.split("_")[0]) for stay in ehr_train.stay.values]
val_subject_ids = [int(stay.split("_")[0]) for stay in ehr_split_val.stay.values]
test_subject_ids = [int(stay.split("_")[0]) for stay in ehr_split_test.stay.values]

all_subject_ids = np.unique(train_subject_ids + val_subject_ids + test_subject_ids)

cxr_splits = cxr_splits[cxr_splits.subject_id.isin(all_subject_ids)]

cxr_splits.loc[cxr_splits.subject_id.isin(train_subject_ids), "split"] = "train"
cxr_splits.loc[cxr_splits.subject_id.isin(val_subject_ids), "split"] = "validate"
cxr_splits.loc[cxr_splits.subject_id.isin(test_subject_ids), "split"] = "test"
print(f"after merge ehr {cxr_splits.split.value_counts()}")

cxr_splits.to_csv(f"{cxr_data_resize_dir}/mimic-cxr-note-ehr-split.csv", index=False)
"""
before update, cxr image 
No: split
train       368960
test          5159
validate      2991
Name: count, dtype: int64
before update, cxr note No: 227781
after merge note, cxr image 
No: split
train       368875
test          5159
validate      2990
Name: count, dtype: int64
after merge ehr split
train       80080
test        23067
validate     9494
Name: count, dtype: int64
"""
