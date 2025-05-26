import pandas as pd

seed = 42

if __name__ == "__main__":
    stay_set = set()
    for task in ["in-hospital-mortality", "readmission"]:
        for split in ["train", "val", "test"]:
            file_path = f"MedFuse/mimic3extract/data/{task}/{split}_listfile.csv"
            data = pd.read_csv(file_path)
            sorted_data = data.sort_values(by="stay", ascending=True)
            print(f"Number of stays in {task} {split}:", len(sorted_data))
            sampled_data = sorted_data.sample(frac=1 / 3, random_state=seed)
            print(
                f"Number of stays in {task} {split} after sampling:", len(sampled_data)
            )
            stay_set.update(sampled_data["stay"])
            print('Number of unique stays:', len(stay_set))
            sampled_data.to_csv(f'MedFuse/mimic3extract/data/{task}/{split}_note_listfile.csv', index=False)


"""
Number of stays in in-hospital-mortality 
    train: 14681
    train after sampling: 4894
    val: 3222
    val after sampling: 1074
    test: 3236
    test after sampling: 1079
    total: 14681+3222+3236 = 21139

Number of stays in readmission 
    train: 14681
    train after sampling: 4894
    val: 3222
    val after sampling: 1074
    test: 3236
    test after sampling: 1079

Number of stays in phenotyping_48h 
    train: 14681
    train after sampling: 4894
    val: 3222
    val after sampling: 1074
    test: 3236
    test after sampling: 1079
"""
