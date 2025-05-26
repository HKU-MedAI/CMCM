import os
import argparse
import numpy as np
import pandas as pd
import random
random.seed(49297)
from tqdm import tqdm


def process_partition(args, partition, eps=1e-6, n_hours=48):
    output_dir = os.path.join(args.output_path, partition)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    xy_pairs = []
    patients = list(filter(str.isdigit, os.listdir(os.path.join(args.root_path, partition))))
    for patient in tqdm(patients, desc='Iterating over patients in {}'.format(partition)):
        patient_folder = os.path.join(args.root_path, partition, patient)
        patient_ts_files = sorted(list(filter(lambda x: x.find("timeseries") != -1, os.listdir(patient_folder))))
        stay_df = pd.read_csv(os.path.join(patient_folder, "stays.csv"))
        stay_df.sort_values(by=['ADMITTIME', 'DISCHTIME'], inplace=True)
        stay_df['readmission_30d'] = np.nan
        for idx, row in stay_df.iterrows():
            if idx == len(stay_df) - 1:
                stay_df.loc[idx, 'readmission_30d'] = 0
                continue
            next_row = stay_df.iloc[idx+1]
            next_intime = pd.to_datetime(next_row.ADMITTIME)
            cur_outtime = pd.to_datetime(row.DISCHTIME)
            if (next_intime - cur_outtime).days < 30:
                stay_df.loc[idx, 'readmission_30d'] = 1
            else:
                stay_df.loc[idx, 'readmission_30d'] = 0

        for ts_filename in patient_ts_files:
            with open(os.path.join(patient_folder, ts_filename)) as tsfile:
                lb_filename = ts_filename.replace("_timeseries", "")
                label_df = pd.read_csv(os.path.join(patient_folder, lb_filename))

                # empty label file
                if label_df.shape[0] == 0:
                    continue
                
                icustay = int(label_df.iloc[0]["Icustay"])
                readmission_30d = int(stay_df[stay_df.ICUSTAY_ID == icustay].iloc[0].readmission_30d)

                mortality = int(label_df.iloc[0]["Mortality"])
                if mortality == 1:
                    readmission_30d = 1

                los = 24.0 * label_df.iloc[0]['Length of Stay']  # in hours
                if pd.isnull(los):
                    print("\n\t(length of stay is missing)", patient, ts_filename)
                    continue

                if los < n_hours - eps:
                    continue

                ts_lines = tsfile.readlines()
                header = ts_lines[0]
                ts_lines = ts_lines[1:]
                event_times = [float(line.split(',')[0]) for line in ts_lines]

                ts_lines = [line for (line, t) in zip(ts_lines, event_times)
                            if -eps < t < n_hours + eps]

                # no measurements in ICU
                if len(ts_lines) == 0:
                    print("\n\t(no events in ICU) ", patient, ts_filename)
                    continue

                output_ts_filename = patient + "_" + ts_filename
                with open(os.path.join(output_dir, output_ts_filename), "w") as outfile:
                    outfile.write(header)
                    for line in ts_lines:
                        outfile.write(line)

                xy_pairs.append((output_ts_filename, icustay, readmission_30d))


    print("Number of created samples:", len(xy_pairs))
    if partition == "train":
        random.shuffle(xy_pairs)
    if partition == "test":
        xy_pairs = sorted(xy_pairs)

    with open(os.path.join(output_dir, "listfile.csv"), "w") as listfile:
        listfile.write('stay,period_length,stay_id,y_true\n')
        for (x, icustay, y) in xy_pairs:
            listfile.write('{},0,{},{:d}\n'.format(x, icustay, y))


def main():
    parser = argparse.ArgumentParser(description="Create data for readmission-30d prediction task.")
    parser.add_argument('--root_path', type=str, 
                        default=os.path.join(os.path.dirname(__file__), '/data/root/'),
                        help="Path to root folder containing train and test sets.")
    parser.add_argument('--output_path', type=str, 
                        default=os.path.join(os.path.dirname(__file__), '/data/readm/'),
                        help="Directory where the created data should be stored.")
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    process_partition(args, "test")
    process_partition(args, "train")


if __name__ == '__main__':
    main()
