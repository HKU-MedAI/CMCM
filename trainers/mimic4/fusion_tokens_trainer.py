from __future__ import absolute_import
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.mimic4.fusion_tokens import FusionTokens
from models.ehr_models import LSTM
from models.cxr_models import CXRModels
from trainers.trainer import Trainer
import pandas as pd
import numpy as np
from sklearn import metrics


class FusionTokensTrainer(Trainer):
    def __init__(self, train_dl, val_dl, args, test_dl=None):

        super(FusionTokensTrainer, self).__init__(args)
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

        self.model = FusionTokens(args, self.ehr_model, self.cxr_model).to(self.device)
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
        # self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.99)
        self.epochs_stats = {
            "loss train": [],
            "loss val": [],
            "auroc val": [],
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
        if self.args.load_state_cxr is not None:
            self.load_cxr_pheno(load_state=self.args.load_state_cxr)

        if self.args.load_state is not None:
            self.load_state()

        if "uni_ehr" in self.args.fusion_type:
            self.freeze(self.model.cxr_model)
        elif "uni_cxr" in self.args.fusion_type:
            self.freeze(self.model.ehr_model)
        elif "late" in self.args.fusion_type:
            self.freeze(self.model)
        elif "early" in self.args.fusion_type:
            self.freeze(self.model.cxr_model)
            self.freeze(self.model.ehr_model)
        elif "lstm" in self.args.fusion_type:
            # self.freeze(self.model.cxr_model)
            # self.freeze(self.model.ehr_model)
            pass

    def train_epoch(self):
        print(f"starting train epoch {self.epoch}")
        epoch_loss = 0
        epoch_loss_align = 0
        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)
        steps = len(self.train_dl)
        for i, (x, img, y_ehr, y_cxr, seq_lengths, pairs) in enumerate(self.train_dl):
            y = self.get_gt(y_ehr, y_cxr)
            x = torch.from_numpy(x).float()
            x = x.to(self.device)
            y = y.to(self.device)
            img = img.to(self.device)

            output = self.model(x, seq_lengths, img, pairs)

            pred = output[self.args.fusion_type].squeeze()
            loss = self.loss(pred, y)
            epoch_loss += loss.item()
            if self.args.align > 0.0:
                loss = loss + self.args.align * output["align_loss"]
                epoch_loss_align = (
                    epoch_loss_align + self.args.align * output["align_loss"].item()
                )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            outPRED = torch.cat((outPRED, pred), 0)
            outGT = torch.cat((outGT, y), 0)

            if i % 100 == 9:
                eta = self.get_eta(self.epoch, i)
                print(
                    f" epoch [{self.epoch:04d} / {self.args.epochs:04d}] [{i:04}/{steps}] eta: {eta:<20}  lr: \t{self.optimizer.param_groups[0]['lr']:0.4E} loss: \t{epoch_loss/i:0.5f} loss align {epoch_loss_align/i:0.4f}"
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
        # ehr_features = torch.FloatTensor()
        # cxr_features = torch.FloatTensor()
        outGT = torch.FloatTensor().to(self.device)
        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)

        with torch.no_grad():
            for i, (x, img, y_ehr, y_cxr, seq_lengths, pairs) in enumerate(dl):
                y = self.get_gt(y_ehr, y_cxr)

                x = torch.from_numpy(x).float()
                x = Variable(x.to(self.device), requires_grad=False)
                y = Variable(y.to(self.device), requires_grad=False)
                img = img.to(self.device)
                output = self.model(x, seq_lengths, img, pairs)

                pred = output[self.args.fusion_type]
                if len(pred.shape) > 1:
                    pred = pred.squeeze()
                    # import pdb; pdb.set_trace()
                # .squeeze()
                loss = self.loss(pred, y)
                epoch_loss += loss.item()
                if self.args.align > 0.0:

                    epoch_loss_align += output["align_loss"].item()
                outPRED = torch.cat((outPRED, pred), 0)
                outGT = torch.cat((outGT, y), 0)
                # if 'ehr_feats' in output:
                #     ehr_features = torch.cat((ehr_features, output['ehr_feats'].data.cpu()), 0)
                # if 'cxr_feats' in output:
                #     cxr_features = torch.cat((cxr_features, output['cxr_feats'].data.cpu()), 0)

        self.scheduler.step(epoch_loss / len(self.val_dl))

        print(
            f"val [{self.epoch:04d} / {self.args.epochs:04d}] validation loss: \t{epoch_loss/i:0.5f} \t{epoch_loss_align/i:0.5f}"
        )
        ret = self.computeAUROC(
            outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), "validation"
        )
        np.save(f"{self.args.save_dir}/pred.npy", outPRED.data.cpu().numpy())
        np.save(f"{self.args.save_dir}/gt.npy", outGT.data.cpu().numpy())

        # if 'ehr_feats' in output:
        #     np.save(f'{self.args.save_dir}/ehr_features.npy', ehr_features.data.cpu().numpy())
        # if 'cxr_feats' in output:
        #     np.save(f'{self.args.save_dir}/cxr_features.npy', cxr_features.data.cpu().numpy())

        self.epochs_stats["auroc val"].append(ret["auroc_mean"])

        self.epochs_stats["loss val"].append(epoch_loss / i)
        self.epochs_stats["loss align val"].append(epoch_loss_align / i)
        # print(f'true {outGT.data.cpu().numpy().sum()}/{outGT.data.cpu().numpy().shape}')
        # print(f'true {outGT.data.cpu().numpy().sum()/outGT.data.cpu().numpy().shape[0]} ({outGT.data.cpu().numpy().sum()}/{outGT.data.cpu().numpy().shape[0]})')

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
       