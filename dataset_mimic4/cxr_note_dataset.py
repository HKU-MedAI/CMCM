from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
import glob
import torchvision.transforms as transforms
from transformers import AutoTokenizer


class MIMIC_CXR_NOTE(Dataset):
    def __init__(
        self,
        paths,
        args,
        transform=None,
        split="train",
        bert_type="huawei-noah/TinyBERT_General_4L_312D",
    ):
        self.data_dir = args.cxr_data_dir
        self.args = args
        self.CLASSES = [
            "Atelectasis",
            "Cardiomegaly",
            "Consolidation",
            "Edema",
            "Enlarged Cardiomediastinum",
            "Fracture",
            "Lung Lesion",
            "Lung Opacity",
            "No Finding",
            "Pleural Effusion",
            "Pleural Other",
            "Pneumonia",
            "Pneumothorax",
            "Support Devices",
        ]
        self.SECTIONS = ["impression", "findings", "last_paragraph", "comparison"]

        self.filenames_to_path = {
            path.split("/")[-1].split(".")[0]: path for path in paths
        }

        metadata = pd.read_csv(f"{self.data_dir}/mimic-cxr-2.0.0-metadata.csv")

        labels = pd.read_csv(f"{self.data_dir}/mimic-cxr-2.0.0-chexpert.csv")
        labels[self.CLASSES] = labels[self.CLASSES].fillna(0)
        labels = labels.replace(-1.0, 0.0)

        splits = pd.read_csv(f"{self.data_dir}/mimic-cxr-note-ehr-split.csv")
        splits[self.SECTIONS] = splits[self.SECTIONS].fillna("")
        metadata_with_labels = metadata.merge(
            labels[self.CLASSES + ["study_id"]], how="inner", on="study_id"
        )

        self.filesnames_to_labels = dict(
            zip(
                metadata_with_labels["dicom_id"].values,
                metadata_with_labels[self.CLASSES].values,
            )
        )
        self.filesnames_to_note_section = dict(
            zip(splits["dicom_id"].values, splits[self.SECTIONS].values)
        )
        self.filenames_loaded = splits.loc[splits.split == split]["dicom_id"].values

        assert set(self.filenames_loaded).issubset(
            self.filesnames_to_note_section.keys()
        )
        self.transform = transform
        self.filenames_loaded = [
            filename
            for filename in self.filenames_loaded
            if filename in self.filesnames_to_labels
        ]

        self.bert_type = bert_type
        self.tokenizer = AutoTokenizer.from_pretrained(bert_type)

        print(f"Loaded {len(self.filenames_loaded)} {split} images with notes")

    def __getitem__(self, index):
        if isinstance(index, str):
            img = Image.open(self.filenames_to_path[index]).convert("RGB")
            note_sections = self.filesnames_to_note_section[index]
            text_token = []
            atten_mask = []
            for t in note_sections:
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

            labels = torch.tensor(self.filesnames_to_labels[index]).float()

            if self.transform is not None:
                img = self.transform(img)
            return img, text_token, atten_mask, labels

        filename = self.filenames_loaded[index]
        img = Image.open(self.filenames_to_path[filename]).convert("RGB")
        note_sections = self.filesnames_to_note_section[filename]
        text_token = []
        atten_mask = []
        for t in note_sections:
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
                attention_mask[0] += 1
                atten_mask.append(attention_mask)
            else:
                atten_mask.append(attention_mask)
        text_token = torch.tensor(text_token, dtype=torch.long)
        atten_mask = torch.tensor(atten_mask, dtype=torch.long)

        labels = torch.tensor(self.filesnames_to_labels[filename]).float()
        if self.transform is not None:
            img = self.transform(img)
        return img, text_token, atten_mask, labels

    def __len__(self):
        return len(self.filenames_loaded)


def get_transforms(args):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_transforms = []
    train_transforms.append(transforms.Resize(256))
    train_transforms.append(transforms.RandomHorizontalFlip())
    train_transforms.append(
        transforms.RandomAffine(
            degrees=45, scale=(0.85, 1.15), shear=0, translate=(0.15, 0.15)
        )
    )
    train_transforms.append(transforms.CenterCrop(224))
    train_transforms.append(transforms.ToTensor())
    train_transforms.append(normalize)

    test_transforms = []
    test_transforms.append(transforms.Resize(args.resize))
    test_transforms.append(transforms.CenterCrop(args.crop))
    test_transforms.append(transforms.ToTensor())
    test_transforms.append(normalize)

    return train_transforms, test_transforms


def get_cxr_note_datasets(args):
    train_transforms, test_transforms = get_transforms(args)
    data_dir = args.cxr_data_dir
    paths = glob.glob(f"{data_dir}/resized/**/*.jpg", recursive=True)
    dataset_train = MIMIC_CXR_NOTE(
        paths, args, split="train", transform=transforms.Compose(train_transforms)
    )
    dataset_validate = MIMIC_CXR_NOTE(
        paths,
        args,
        split="validate",
        transform=transforms.Compose(test_transforms),
    )
    dataset_test = MIMIC_CXR_NOTE(
        paths,
        args,
        split="test",
        transform=transforms.Compose(test_transforms),
    )

    return dataset_train, dataset_validate, dataset_test
