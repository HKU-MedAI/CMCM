import os
import pickle
import numpy as np
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def lookup(w2i_lookup, x):
    if x in w2i_lookup:
        return w2i_lookup[x]
    else:
        return len(w2i_lookup)


def diff(time1, time2):
    # compute time2-time1
    # return difference in hours
    a = np.datetime64(time1)
    b = np.datetime64(time2)
    h = (b - a).astype("timedelta64[h]").astype(int)
    """if h < -1e-6:
        print(h)
        assert h > 1e-6"""
    return h


class MIMICNOTE(Dataset):
    def __init__(
        self,
        dataset_dir,
        bert_type="yikuan8/Clinical-Longformer",
        task="in-hospital-mortality",
        split="train",
        return_names=True,
        period_length=48.0,
    ):
        self.return_names = return_names
        self._period_length = period_length
        self._dataset_dir = dataset_dir
        listfile_path = os.path.join(dataset_dir, f"{task}/{split}_note_listfile.csv")
        with open(listfile_path, "r") as lfile:
            self._data = lfile.readlines()
        self._listfile_header = self._data[0]
        self.CLASSES = self._listfile_header.strip().split(",")[3:]
        self._data = self._data[1:]
        self._data = [line.split(",") for line in self._data]
        self.data_map = {
            mas[0]: {
                "labels": list(map(float, mas[3:])),
                "stay_id": float(mas[2]),
                "time": float(mas[1]),
            }
            for mas in self._data
        }
        self.names = list(self.data_map.keys())

        if (split == "train") or (split == "val"):
            self.note_path = os.path.join(dataset_dir, "train_text_fixed")
            self.starttime_path = os.path.join(dataset_dir, "train_starttime.pkl")
        elif split == "test":
            self.note_path = os.path.join(dataset_dir, "test_text_fixed")
            self.starttime_path = os.path.join(dataset_dir, "test_starttime.pkl")
        self.all_files = set(os.listdir(self.note_path))
        with open(self.starttime_path, "rb") as f:
            self.episodeToStartTime = pickle.load(f)

        data_text, data_times, data_time, filtered_names = (
            self.read_all_text_append_json(
                self.names, self._period_length, NumOfNotes=5, notes_aggeregate="Last"
            )
        )
        self.data_text = data_text
        self.filtered_names = filtered_names
        self.bert_type = bert_type
        self.tokenizer = AutoTokenizer.from_pretrained(bert_type)
        print(f"Number of patients in {split} split: {len(self.filtered_names)}")

    def get_name_from_filename(self, fname):
        # '24610_episode1_timeseries.csv'
        tokens = fname.split("_")
        pid = tokens[0]
        episode_id = tokens[1].replace("episode", "").strip()
        return pid, episode_id

    def read_text_event_json(self, text_file_name):
        filepath = os.path.join(self.note_path, str(text_file_name))
        with open(filepath, "r") as f:
            d = json.load(f)
        time = sorted(d.keys())
        text = []
        for t in time:
            text.append(" ".join(d[t]))
        assert len(time) == len(text)
        return time, text

    def read_all_text_append_json(
        self, names, period_length=48.0, NumOfNotes=5, notes_aggeregate="Last"
    ):
        texts_dict = {}
        time_dict = {}
        start_times = {}
        filtered_names = []
        for patient_id in names:
            pid, eid = self.get_name_from_filename(patient_id)
            text_file_name = pid + "_" + eid
            if text_file_name in self.all_files:
                time, texts = self.read_text_event_json(text_file_name)
                start_time = self.episodeToStartTime[text_file_name]
                if len(texts) == 0 or start_time == -1:
                    continue
                final_concatenated_text = []
                times_array = []
                for t, txt in zip(time, texts):
                    # and  diff(start_time, t)>=(-24-1e-6)
                    if diff(start_time, t) <= period_length + 1e-6:
                        final_concatenated_text.append(txt)
                        times_array.append(t)
                    else:
                        break
                # if length of notes is less than NumOfNotes, skip this patient
                if len(final_concatenated_text) <= NumOfNotes:
                    continue
                if notes_aggeregate == "First":
                    texts_dict[patient_id] = final_concatenated_text[:NumOfNotes]
                    time_dict[patient_id] = times_array[:NumOfNotes]
                else:
                    texts_dict[patient_id] = final_concatenated_text[-NumOfNotes:]
                    time_dict[patient_id] = times_array[-NumOfNotes:]
                start_times[patient_id] = start_time
                filtered_names.append(patient_id)
        return texts_dict, time_dict, start_times, filtered_names

    def __getitem__(self, index):
        if isinstance(index, int):
            index = self.filtered_names[index]
        label = self.data_map[index]["labels"]
        text_token = []
        atten_mask = []
        text = self.data_text[index]
        for t in text:
            inputs = self.tokenizer.encode_plus(
                t,
                padding="max_length",
                max_length=512,
                add_special_tokens=True,
                return_attention_mask=True,
                truncation=True,
            )
            text_token.append(inputs["input_ids"])
            attention_mask = inputs["attention_mask"]
            if "Longformer" in self.bert_type:
                attention_mask[0] += 1  # type: ignore
                atten_mask.append(attention_mask)
            else:
                atten_mask.append(attention_mask)
        text_token = torch.tensor(text_token, dtype=torch.long)
        atten_mask = torch.tensor(atten_mask, dtype=torch.long)
        label = (
            torch.tensor(label, dtype=torch.float)
            if len(label) > 1
            else torch.tensor(label, dtype=torch.float)[0]
        )

        return (
            text_token,
            atten_mask,
            label,
        )  # bs * num_notes * max_length, bs * num_notes * max_length, bs * num_labels

    def __len__(self):
        return len(self.filtered_names)


def get_note_datasets(args):
    data_dir = args.ehr_data_dir
    dataset_train = MIMICNOTE(
        data_dir, bert_type=args.bert_type, task=args.task, split="train"
    )
    dataset_validate = MIMICNOTE(
        data_dir, bert_type=args.bert_type, task=args.task, split="val"
    )
    dataset_test = MIMICNOTE(
        data_dir, bert_type=args.bert_type, task=args.task, split="test"
    )

    return dataset_train, dataset_validate, dataset_test