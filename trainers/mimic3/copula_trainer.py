from __future__ import absolute_import
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.mimic3.fusion_copula import Copula_Fusion
from models.ehr_models import LSTM
from models.note_models import BertForRepresentation as NoteModels
from trainers.trainer import Trainer
import wandb
import numpy as np


class CopulaTrainer(Trainer):
    def __init__(self, train_dl, val_dl, args, test_dl=None):

        super(CopulaTrainer, self).__init__(args)
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
        self.note_model = NoteModels(self.args).to(self.device)

        self.model = Copula_Fusion(args, self.ehr_model, self.note_model).to(
            self.device
        )
        self.init_fusion_method()

        self.loss = nn.BCELoss()

        self.optimizer = optim.Adam(
            self.model.parameters(), args.lr, betas=(0.9, self.args.beta_1)
        )
        self.load_state()

        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"model size: {num_params}")
        print(self.ehr_model)
        print(self.note_model)
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
            "auroc test": [],
            "loss copula train": [],
            "loss copula val": [],
            "loss align train": [],
            "loss align val": [],
        }

    def init_fusion_method(self):
        """
        for early fusion
        load pretrained encoders and
        freeze both encoders
        """

        if self.args.load_state_ehr is not None:
            self.load_ehr_pheno(load_state=self.args.load_state_ehr)
        if self.args.load_state_note is not None:
            self.load_note_pheno(load_state=self.args.load_state_note)
        if self.args.load_state is not None:
            self.load_state()

    def train_epoch(self):
        print(f"starting train epoch {self.epoch}")
        epoch_loss = 0
        epoch_loss_copula = 0

        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)
        steps = len(self.train_dl)
        for i, (x, token, mask, y_ehr, y_note, seq_lengths, pairs) in enumerate(
            self.train_dl
        ):
            y = self.get_gt_mimic3(y_ehr, y_note)
            x = torch.from_numpy(x).float()
            x = x.to(self.device)
            y = y.to(self.device)
            token = token.to(self.device)
            mask = mask.to(self.device)
            if (
                self.args.task == "in-hospital-mortality"
                or self.args.task == "readmission"
            ):
                y = y.unsqueeze(1)

            output = self.model(x, seq_lengths, token, mask, pairs)

            pred = output[self.args.fusion_type]
            loss = self.loss(pred, y)

            epoch_loss += loss.item()
            loss = loss + self.args.copula * output["copula_loss"]
            epoch_loss_copula += self.args.copula * output["copula_loss"].item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            outPRED = torch.cat((outPRED, pred), 0)
            outGT = torch.cat((outGT, y), 0)

            if i % 100 == 9:
                eta = self.get_eta(self.epoch, i)
                print(
                    f" epoch [{self.epoch:04d} / {self.args.epochs:04d}] [{i:04}/{steps}] eta: {eta:<20}  lr: \t{self.optimizer.param_groups[0]['lr']:0.4E} \tloss: {epoch_loss/i:0.5f}, loss copula: {epoch_loss_copula/i:0.4f}"
                )

        ret = self.computeAUROC(
            outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), "train"
        )
        self.epochs_stats["loss train"].append(epoch_loss / i)
        self.epochs_stats["loss copula train"].append(epoch_loss_copula / i)
        if wandb.run is not None:
            wandb.log(
                {
                    "loss": epoch_loss / i,
                    "loss_copula": epoch_loss_copula / i,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "train_auroc": ret["auroc_mean"],
                    "trainauprc": ret["auprc_mean"],
                    "theta": self.model.copula_loss.theta.item(),
                }
            )

        return ret

    def validate(self, dl):
        print(f"starting val epoch {self.epoch}")
        epoch_loss = 0
        epoch_loss_copula = 0
        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)

        with torch.no_grad():
            for i, (x, token, mask, y_ehr, y_note, seq_lengths, pairs) in enumerate(dl):
                y = self.get_gt_mimic3(y_ehr, y_note)
                x = torch.from_numpy(x).float()
                x = Variable(x.to(self.device), requires_grad=False)
                y = Variable(y.to(self.device), requires_grad=False)
                token = token.to(self.device)
                mask = mask.to(self.device)

                if (
                    self.args.task == "in-hospital-mortality"
                    or self.args.task == "readmission"
                ):
                    y = y.unsqueeze(1)

                output = self.model(x, seq_lengths, token, mask, pairs)
                pred = output[self.args.fusion_type]
                loss = self.loss(pred, y)
                epoch_loss += loss.item()
                loss += self.args.copula * output["copula_loss"]
                epoch_loss_copula += self.args.copula * output["copula_loss"].item()
                outPRED = torch.cat((outPRED, pred), 0)
                outGT = torch.cat((outGT, y), 0)

        self.scheduler.step(epoch_loss / len(self.val_dl))

        print(
            f"val [{self.epoch:04d} / {self.args.epochs:04d}] validation loss: \t{epoch_loss/i:0.5f}, validation copula loss: {epoch_loss_copula/i:0.5f}"
        )
        ret = self.computeAUROC(
            outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), "validation"
        )
        np.save(f"{self.args.save_dir}/pred.npy", outPRED.data.cpu().numpy())
        np.save(f"{self.args.save_dir}/gt.npy", outGT.data.cpu().numpy())

        self.epochs_stats["auroc val"].append(ret["auroc_mean"])
        self.epochs_stats["loss val"].append(epoch_loss / i)
        self.epochs_stats["loss copula val"].append(epoch_loss_copula / i)

        if wandb.run is not None:
            wandb.log(
                {
                    "val_auroc": ret["auroc_mean"],
                    "val_auprc": ret["auprc_mean"],
                    "val_auroc_ci_l": ret["ci_auroc"][0][0],
                    "val_auprc_ci_l": ret["ci_auprc"][0][0],
                    "val_auroc_ci_u": ret["ci_auroc"][0][1],
                    "val_auprc_ci_u": ret["ci_auprc"][0][1],
                }
            )

        return ret

    def quick_test(self, dl):
        print(f"starting quick test epoch {self.epoch}")
        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)

        with torch.no_grad():
            for i, (x, token, mask, y_ehr, y_note, seq_lengths, pairs) in enumerate(dl):
                y = self.get_gt_mimic3(y_ehr, y_note)
                x = torch.from_numpy(x).float()
                x = Variable(x.to(self.device), requires_grad=False)
                y = Variable(y.to(self.device), requires_grad=False)
                token = token.to(self.device)
                mask = mask.to(self.device)

                if (
                    self.args.task == "in-hospital-mortality"
                    or self.args.task == "readmission"
                ):
                    y = y.unsqueeze(1)

                output = self.model(x, seq_lengths, token, mask, pairs)
                pred = output[self.args.fusion_type]
                loss = self.loss(pred, y)
                loss += self.args.copula * output["copula_loss"]
                outPRED = torch.cat((outPRED, pred), 0)
                outGT = torch.cat((outGT, y), 0)

        print(f"test [{self.epoch:04d} / {self.args.epochs:04d}]")
        ret = self.computeAUROC(
            outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), "test"
        )
        np.save(f"{self.args.save_dir}/pred.npy", outPRED.data.cpu().numpy())
        np.save(f"{self.args.save_dir}/gt.npy", outGT.data.cpu().numpy())

        self.epochs_stats["auroc test"].append(ret["auroc_mean"])

        if wandb.run is not None:
            wandb.log(
                {
                    "test_auroc": ret["auroc_mean"],
                    "test_auprc": ret["auprc_mean"],
                    "test_auroc_ci_l": ret["ci_auroc"][0][0],
                    "test_auprc_ci_l": ret["ci_auprc"][0][0],
                    "test_auroc_ci_u": ret["ci_auroc"][0][1],
                    "test_auprc_ci_u": ret["ci_auprc"][0][1],
                }
            )

        return ret

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
            self.quick_test(self.test_dl)
            self.save_checkpoint(prefix="last")

            if self.best_auroc < ret["auroc_mean"]:
                self.best_auroc = ret["auroc_mean"]
                self.best_stats = ret
                self.save_checkpoint(prefix="best")
                # print(f'saving best AUROC {ret["ave_auc_micro"]:0.4f} checkpoint')
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
