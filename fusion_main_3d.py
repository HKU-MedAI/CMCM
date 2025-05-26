from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import os
import random
from trainers.triple.fusion_trainer import FusionTrainer
from trainers.triple.mmtm_trainer import MMTMTrainer
from trainers.triple.daft_trainer import DAFTTrainer
from trainers.triple.drfuse_trainer import DrFuseTrainer
from trainers.triple.copula_trainer import CopulaTrainer
from ehr_utils.preprocessing import Discretizer, Normalizer
from dataset_mimic4.ehr_dataset import get_datasets
from dataset_mimic4.cxr_note_dataset import get_cxr_note_datasets
from dataset_mimic4.fusion_3d import load_cxr_note_ehr
from pathlib import Path
from paths import *
import torch
from arguments import args_parser


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"set seed {seed}")


def read_timeseries(args):
    path = f"{args.ehr_data_dir}/{args.task}/train/14991576_episode3_timeseries.csv"
    ret = []
    with open(path, "r") as tsfile:
        header = tsfile.readline().strip().split(",")
        assert header[0] == "Hours"
        for line in tsfile:
            mas = line.strip().split(",")
            ret.append(np.array(mas))
    return np.stack(ret)


def get_ehr_dataset_info(args):
    discretizer = Discretizer(
        timestep=float(args.timestep),
        store_masks=True,
        impute_strategy="previous",
        start_time="zero",
    )

    discretizer_header = discretizer.transform(read_timeseries(args))[1].split(",")
    cont_channels = [
        i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1
    ]

    normalizer = Normalizer(
        fields=cont_channels
    )  # choose here which columns to standardize
    normalizer_state = args.normalizer_state
    if normalizer_state is None:
        normalizer_state = (
            "normalizers/ph_ts{}.input_str:previous.start_time:zero.normalizer".format(
                args.timestep
            )
        )
        normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
    normalizer.load_params(normalizer_state)
    return discretizer, normalizer


if __name__ == "__main__":
    parser = args_parser()
    # add more arguments here ...
    args = parser.parse_args()
    args.ehr_data_dir = MIMIC4_DATA_DIR
    args.cxr_data_dir = CXR_DATA_DIR
    args.normalizer_state = MIMIC4_READM_NORMALIZER_PATH if args.task == "readmission" else MIMIC4_IHM_NORMALIZER_PATH
    args.data_pairs = "paired_ehr_cxr"

    path = Path(args.save_dir)
    path.mkdir(parents=True, exist_ok=True)

    discretizer, normalizer = get_ehr_dataset_info(args)

    for seed in [1002]:
        args.load_state = None
        print(args)

        # set seed
        seed_torch(seed)

        ehr_train_ds, ehr_val_ds, ehr_test_ds = get_datasets(
            discretizer, normalizer, args
        )

        cxr_train_ds, cxr_val_ds, cxr_test_ds = get_cxr_note_datasets(args)

        train_dl, val_dl, test_dl = load_cxr_note_ehr(
            args,
            ehr_train_ds,
            ehr_val_ds,
            cxr_train_ds,
            cxr_val_ds,
            ehr_test_ds,
            cxr_test_ds,
        )

        with open(f"{args.save_dir}/args.txt", "w") as results_file:
            for arg in vars(args):
                print(f"  {arg:<40}: {getattr(args, arg)}")
                results_file.write(f"  {arg:<40}: {getattr(args, arg)}\n")

        if args.fusion_type == "mmtm":
            trainer = MMTMTrainer(train_dl, val_dl, args, test_dl=test_dl)
        elif args.fusion_type == "daft":
            trainer = DAFTTrainer(train_dl, val_dl, args, test_dl=test_dl)
        elif args.fusion_type == "drfuse":
            trainer = DrFuseTrainer(train_dl, val_dl, args, test_dl=test_dl)
        elif args.fusion_type == "copula":
            trainer = CopulaTrainer(train_dl, val_dl, args, test_dl=test_dl)
        else:
            trainer = FusionTrainer(train_dl, val_dl, args, test_dl=test_dl)

        if args.mode == "train":
            print("==> training")
            trainer.train()
            trainer.args.load_state = args.save_dir + "/best_checkpoint.pth.tar"
            trainer.load_state()
            trainer.eval()

        elif args.mode == "eval":
            trainer.args.load_state = args.save_dir + "/best_checkpoint.pth.tar"
            trainer.load_state()
            trainer.eval()

        else:
            raise ValueError("not Implementation for args.mode")

