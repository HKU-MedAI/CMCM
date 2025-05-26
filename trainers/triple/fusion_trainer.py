from __future__ import absolute_import
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.triple.fusion import Fusion
from models.ehr_models import LSTM
from models.cxr_models import CXRModels
from models.note_models import BertForRepresentation as NoteModels
from trainers.trainer import Trainer
import pandas as pd
import numpy as np
from sklearn import metrics


class FusionTrainer(Trainer):
    def __init__(self, train_dl, val_dl, args, test_dl=None):

        super(FusionTrainer, self).__init__(args)
        self.epoch = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.args = args
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl

        self.ehr_model = LSTM(
            input_dim=76,
            num_classes=args.num_classes,
            hidden_dim=args.dim,
            dropout=args.dropout,
            layers=args.layers,
        ).to(self.device)
        self.cxr_model = CXRModels(self.args, self.device).to(self.device)
        self.note_model = NoteModels(self.args).to(self.device)

        self.model = Fusion(args, self.ehr_model, self.cxr_model, self.note_model).to(
            self.device
        )
        self.init_fusion_method()

        self.loss = nn.BCELoss()

        self.optimizer = optim.Adam(
            self.model.parameters(), args.lr, betas=(0.9, self.args.beta_1)
        )
        self.load_state()
        print(self.ehr_model)
        print(self.optimizer)
        print(self.loss)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, factor=0.5, patience=10, mode="min"
        )

        self.best_auroc = 0
        self.best_stats = None
        self.epochs_stats = {
            "loss train": [],
            "loss val": [],
            "auroc val": [],
            "loss copula train": [],
            "loss copula val": [],
            "loss align train": [],
            "loss align val": [],
        }

    def init_fusion_method(self):
        if self.args.load_state is not None:
            self.load_state()

    def train_epoch(self):
        print(f"starting train epoch {self.epoch}")
        epoch_loss = 0
        epoch_loss_align = 0

        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)
        steps = len(self.train_dl)
        for i, (x, img, token, mask, y_ehr, y_cxr, seq_lengths, pairs) in enumerate(
            self.train_dl
        ):
            y = self.get_gt(y_ehr, y_cxr)
            x = torch.from_numpy(x).float()
            x = x.to(self.device)
            y = y.to(self.device)
            img = img.to(self.device)
            token = token.to(self.device)
            mask = mask.to(self.device)
            if (
                self.args.task == "in-hospital-mortality"
                or self.args.task == "readmission"
            ):
                y = y.unsqueeze(1)

            output = self.model(x, seq_lengths, img, token, mask, pairs)

            pred = output[self.args.fusion_type]
            loss = self.loss(pred, y)

            epoch_loss += loss.item()

            if self.args.align > 0.0:
                loss += self.args.align * output["align_loss"]
                epoch_loss_align += self.args.align * output["align_loss"].item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            outPRED = torch.cat((outPRED, pred), 0)
            outGT = torch.cat((outGT, y), 0)

            if i % 100 == 9:
                eta = self.get_eta(self.epoch, i)
                print(
                    f" epoch [{self.epoch:04d} / {self.args.epochs:04d}] [{i:04}/{steps}] eta: {eta:<20}  lr: \t{self.optimizer.param_groups[0]['lr']:0.4E} \tloss: {epoch_loss/i:0.5f} loss align: \t{epoch_loss_align/i:0.4f}"
                )
        ret = self.computeAUROC(
            outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), "train"
        )
        self.epochs_stats["loss train"].append(epoch_loss / i)
        self.epochs_stats["loss align train"].append(epoch_loss_align / i)
        return ret

    def validate(self, dl):
        print(f"starting val epoch {self.epoch}")
        epoch_loss = 0
        epoch_loss_align = 0
        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)

        with torch.no_grad():
            for i, (x, img, token, mask, y_ehr, y_cxr, seq_lengths, pairs) in enumerate(
                dl
            ):
                y = self.get_gt(y_ehr, y_cxr)
                x = torch.from_numpy(x).float()
                x = x.to(self.device)
                y = y.to(self.device)
                img = img.to(self.device)
                token = token.to(self.device)
                mask = mask.to(self.device)
                if (
                    self.args.task == "in-hospital-mortality"
                    or self.args.task == "readmission"
                ):
                    y = y.unsqueeze(1)

                output = self.model(x, seq_lengths, img, token, mask, pairs)

                pred = output[self.args.fusion_type]
                loss = self.loss(pred, y)
                epoch_loss += loss.item()
                if self.args.align > 0.0:
                    loss += self.args.align * output["align_loss"]
                    epoch_loss_align += self.args.align * output["align_loss"].item()
                outPRED = torch.cat((outPRED, pred), 0)
                outGT = torch.cat((outGT, y), 0)

        self.scheduler.step(epoch_loss / len(self.val_dl))

        print(
            f"val [{self.epoch:04d} / {self.args.epochs:04d}] validation loss: \t{epoch_loss/i:0.5f} \t{epoch_loss_align/i:0.5f}"
        )
        ret = self.computeAUROC(
            outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), "validation"
        )
        np.save(f"{self.args.save_dir}/pred.npy", outPRED.data.cpu().numpy())
        np.save(f"{self.args.save_dir}/gt.npy", outGT.data.cpu().numpy())

        self.epochs_stats["auroc val"].append(ret["auroc_mean"])
        self.epochs_stats["loss val"].append(epoch_loss / i)
        self.epochs_stats["loss align val"].append(epoch_loss_align / i)

        return ret

    def compute_late_fusion(self, y_true, uniout_cxr, uniout_ehr):
        y_true = np.array(y_true)
        predictions_cxr = np.array(uniout_cxr)
        predictions_ehr = np.array(uniout_ehr)
        best_weights = np.ones(y_true.shape[-1])
        best_auroc = 0.0
        weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        for class_idx in range(y_true.shape[-1]):
            for weight in weights:
                predictions = (predictions_ehr * best_weights) + (
                    predictions_cxr * (1 - best_weights)
                )
                predictions[:, class_idx] = (predictions_ehr[:, class_idx] * weight) + (
                    predictions_cxr[:, class_idx] * 1 - weight
                )
                auc_scores = metrics.roc_auc_score(y_true, predictions, average=None)
                auroc_mean = np.mean(np.array(auc_scores))
                if auroc_mean > best_auroc:
                    best_auroc = auroc_mean
                    best_weights[class_idx] = weight
                # predictions = weight * predictions_cxr[]

        predictions = (predictions_ehr * best_weights) + (
            predictions_cxr * (1 - best_weights)
        )
        print(best_weights)

        auc_scores = metrics.roc_auc_score(y_true, predictions, average=None)
        ave_auc_micro = metrics.roc_auc_score(y_true, predictions, average="micro")
        ave_auc_macro = metrics.roc_auc_score(y_true, predictions, average="macro")
        ave_auc_weighted = metrics.roc_auc_score(
            y_true, predictions, average="weighted"
        )

        best_stats = {
            "auc_scores": auc_scores,
            "ave_auc_micro": ave_auc_micro,
            "ave_auc_macro": ave_auc_macro,
            "ave_auc_weighted": ave_auc_weighted,
            "auroc_mean": np.mean(np.array(auc_scores)),
        }
        self.print_and_write(
            best_stats, isbest=True, prefix="late fusion weighted average"
        )

        return best_stats

    def eval_age(self):

        print("validating ... ")

        patiens = pd.read_csv("data/physionet.org/files/mimic-iv-1.0/core/patients.csv")
        subject_ids = np.array(
            [int(item.split("_")[0]) for item in self.test_dl.dataset.ehr_files_paired]
        )

        selected = patiens[patiens.subject_id.isin(subject_ids)]
        start = 18
        copy_ehr = np.copy(self.test_dl.dataset.ehr_files_paired)
        copy_cxr = np.copy(self.test_dl.dataset.cxr_files_paired)
        self.model.eval()
        step = 20
        for i in range(20, 100, step):
            subjects = selected.loc[
                ((selected.anchor_age >= start) & (selected.anchor_age < i + step))
            ].subject_id.values
            indexes = [
                jj for (jj, subject) in enumerate(subject_ids) if subject in subjects
            ]

            self.test_dl.dataset.ehr_files_paired = copy_ehr[indexes]
            self.test_dl.dataset.cxr_files_paired = copy_cxr[indexes]

            print(len(indexes))
            ret = self.validate(self.test_dl)
            print(
                f"{start}-{i + step} & {len(indexes)} & & & {ret['auroc_mean']:0.3f} & {ret['auprc_mean']:0.3f}"
            )

            self.print_and_write(
                ret,
                isbest=True,
                prefix=f"{self.args.fusion_type} val",
                filename=f"results_test_{start}_{i + step}.txt",
            )
            start = i + step

    def test(self):
        print("validating ... ")
        self.epoch = 0
        self.model.eval()
        ret = self.validate(self.val_dl)
        self.print_and_write(
            ret,
            isbest=True,
            prefix=f"{self.args.fusion_type} val",
            filename="results_val.txt",
        )
        self.model.eval()
        ret = self.validate(self.test_dl)
        self.print_and_write(
            ret,
            isbest=True,
            prefix=f"{self.args.fusion_type} test",
            filename="results_test.txt",
        )
        return

    def eval(self):
        # self.eval_age()
        print("validating ... ")
        self.epoch = 0
        self.model.eval()
        ret = self.validate(self.val_dl)
        self.print_and_write(
            ret,
            isbest=True,
            prefix=f"{self.args.fusion_type} val",
            filename="results_val.txt",
        )
        self.model.eval()
        ret = self.validate(self.test_dl)
        self.print_and_write(
            ret,
            isbest=True,
            prefix=f"{self.args.fusion_type} test",
            filename="results_test.txt",
        )
        return

    def train(self):
        print(f"running for fusion_type {self.args.fusion_type}")
        for self.epoch in range(self.start_epoch, self.args.epochs):
            self.model.eval()
            ret = self.validate(self.val_dl)
            self.save_checkpoint(prefix="last")

            if self.best_auroc < ret["auroc_mean"]:
                self.best_auroc = ret["auroc_mean"]
                self.best_stats = ret
                self.save_checkpoint(prefix="best")
                self.print_and_write(ret, isbest=True)
                self.patience = 0
            else:
                self.print_and_write(ret, isbest=False)
                self.patience += 1

            self.model.train()
            self.train_epoch()
            self.plot_stats(key="loss", filename="loss.pdf")
            self.plot_stats(key="auroc", filename="auroc.pdf")
            if self.patience >= self.args.patience:
                break
        self.print_and_write(self.best_stats, isbest=True)
