import math
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.nn import functional as F
from trainers.trainer import Trainer
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.mimic4.drfuse import DrFuseModel


class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction="none", log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor, masks):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return (
            0.5
            * (self.kl(m, p.log()) + self.kl(m, q.log())).sum()
            / max(1e-6, masks.sum())
        )


class DrFuseTrainer(Trainer):
    def __init__(self, train_dl, val_dl, args, test_dl=None):
        super(DrFuseTrainer, self).__init__(args)
        self.epoch = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.args = args
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl
        self.model = DrFuseModel(
            hidden_size=args.hidden_size,
            num_classes=args.num_classes,
            ehr_dropout=args.dropout,
            ehr_n_head=args.ehr_n_head,
            ehr_n_layers=args.ehr_n_layers,
        ).to(self.device)

        self.pred_criterion = nn.BCELoss(reduction="none")
        self.alignment_cos_sim = nn.CosineSimilarity(dim=1)
        self.triplet_loss = nn.TripletMarginLoss(reduction="none")
        self.mse_loss = nn.MSELoss(reduction="none")
        self.jsd = JSD()

        self.optimizer = optim.Adam(
            self.model.parameters(), args.lr, weight_decay=self.args.wd
        )
        self.load_state()
        print(self.model)
        print(self.optimizer)
        print(self.pred_criterion)

        self.scheduler = ReduceLROnPlateau(
            self.optimizer, factor=0.5, patience=10, mode="min"
        )

        self.best_auroc = 0
        self.best_stats = None
        self.epochs_stats = {"loss train": [], "loss val": [], "auroc val": []}

    def _compute_masked_pred_loss(self, input, target, mask):
        return (self.pred_criterion(input, target).mean(dim=1) * mask).sum() / max(
            mask.sum(), 1e-6
        )

    def _masked_abs_cos_sim(self, x, y, mask):
        return (self.alignment_cos_sim(x, y).abs() * mask).sum() / max(mask.sum(), 1e-6)

    def _disentangle_loss_jsd(self, model_output, pairs, mode="train"):
        ehr_mask = torch.ones_like(pairs)
        loss_sim_cxr = self._masked_abs_cos_sim(
            model_output["feat_cxr_shared"], model_output["feat_cxr_distinct"], pairs
        )
        loss_sim_ehr = self._masked_abs_cos_sim(
            model_output["feat_ehr_shared"], model_output["feat_ehr_distinct"], ehr_mask
        )

        jsd = self.jsd(
            model_output["feat_ehr_shared"].sigmoid(),
            model_output["feat_cxr_shared"].sigmoid(),
            pairs,
        )

        loss_disentanglement = (
            self.args.lambda_disentangle_shared * jsd
            + self.args.lambda_disentangle_ehr * loss_sim_ehr
            + self.args.lambda_disentangle_cxr * loss_sim_cxr
        )

        return loss_disentanglement

    def _compute_prediction_losses(self, model_output, y_gt, pairs, mode="train"):
        ehr_mask = torch.ones_like(model_output["pred_final"][:, 0])
        loss_pred_final = self._compute_masked_pred_loss(
            model_output["pred_final"], y_gt, ehr_mask
        )
        loss_pred_ehr = self._compute_masked_pred_loss(
            model_output["pred_ehr"], y_gt, ehr_mask
        )
        loss_pred_cxr = self._compute_masked_pred_loss(
            model_output["pred_cxr"], y_gt, pairs
        )
        loss_pred_shared = self._compute_masked_pred_loss(
            model_output["pred_shared"], y_gt, ehr_mask
        )

        return loss_pred_final, loss_pred_ehr, loss_pred_cxr, loss_pred_shared

    def _compute_and_log_loss(self, model_output, y_gt, pairs, mode="train"):
        prediction_losses = self._compute_prediction_losses(
            model_output, y_gt, pairs, mode
        )
        loss_pred_final, loss_pred_ehr, loss_pred_cxr, loss_pred_shared = (
            prediction_losses
        )

        loss_prediction = (
            self.args.lambda_pred_shared * loss_pred_shared
            + self.args.lambda_pred_ehr * loss_pred_ehr
            + self.args.lambda_pred_cxr * loss_pred_cxr
        )

        loss_prediction = loss_pred_final + loss_prediction

        loss_disentanglement = self._disentangle_loss_jsd(model_output, pairs, mode)

        loss_total = loss_prediction + loss_disentanglement

        # aux loss for attention ranking
        raw_pred_loss_ehr = F.binary_cross_entropy(
            model_output["pred_ehr"].data, y_gt, reduction="none"
        )
        raw_pred_loss_cxr = F.binary_cross_entropy(
            model_output["pred_cxr"].data, y_gt, reduction="none"
        )
        raw_pred_loss_shared = F.binary_cross_entropy(
            model_output["pred_shared"].data, y_gt, reduction="none"
        )

        pairs = pairs.unsqueeze(1)
        attn_weights = model_output["attn_weights"]
        attn_ehr, attn_shared, attn_cxr = (
            attn_weights[:, :, 0],
            attn_weights[:, :, 1],
            attn_weights[:, :, 2],
        )

        cxr_overweights_ehr = 2 * (raw_pred_loss_cxr < raw_pred_loss_ehr).float() - 1
        loss_attn1 = pairs * F.margin_ranking_loss(
            attn_cxr, attn_ehr, cxr_overweights_ehr, reduction="none"
        )
        loss_attn1 = loss_attn1.sum() / max(1e-6, loss_attn1[loss_attn1 > 0].numel())

        shared_overweights_ehr = (
            2 * (raw_pred_loss_shared < raw_pred_loss_ehr).float() - 1
        )
        loss_attn2 = pairs * F.margin_ranking_loss(
            attn_shared, attn_ehr, shared_overweights_ehr, reduction="none"
        )
        loss_attn2 = loss_attn2.sum() / max(1e-6, loss_attn2[loss_attn2 > 0].numel())

        shared_overweights_cxr = (
            2 * (raw_pred_loss_shared < raw_pred_loss_cxr).float() - 1
        )
        loss_attn3 = pairs * F.margin_ranking_loss(
            attn_shared, attn_cxr, shared_overweights_cxr, reduction="none"
        )
        loss_attn3 = loss_attn3.sum() / max(1e-6, loss_attn3[loss_attn3 > 0].numel())

        loss_attn_ranking = (loss_attn1 + loss_attn2 + loss_attn3) / 3

        loss_total = loss_total + self.args.lambda_attn_aux * loss_attn_ranking

        return loss_total

    def _get_alignment_lambda(self):
        if self.args.adaptive_adc_lambda:
            lmbda = 2 / (1 + math.exp(-self.args.gamma * self.epoch)) - 1
        else:
            lmbda = 1
        return lmbda

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

    def train_epoch(self):
        print(f"starting train epoch {self.epoch}")
        epoch_loss = 0
        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)
        steps = len(self.train_dl)
        for i, (x, img, y_ehr, y_cxr, seq_lengths, pairs) in enumerate(self.train_dl):
            y = self.get_gt(y_ehr, y_cxr)
            x = torch.from_numpy(x).float()
            x = x.to(self.device)
            y = y.to(self.device)
            img = img.to(self.device)
            pairs = torch.FloatTensor(pairs).to(self.device)
            if (
                self.args.task == "in-hospital-mortality"
                or self.args.task == "readmission"
            ):
                y = y.unsqueeze(1)
            out = self.model(x, img, seq_lengths, pairs, self._get_alignment_lambda())
            loss = self._compute_and_log_loss(out, y_gt=y, pairs=pairs)

            epoch_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.args.attn_fusion == "avg":
                perd_final = (
                    out["pred_ehr"] + out["pred_cxr"] + out["pred_shared"]
                ) / 3
                pred_final = (1 - pairs.unsqueeze(1)) * (
                    out["pred_ehr"] + out["pred_shared"]
                ) / 2 + pairs.unsqueeze(1) * perd_final
            else:
                pred_final = out["pred_final"]

            outPRED = torch.cat((outPRED, pred_final), 0)
            outGT = torch.cat((outGT, y), 0)

            if i % 100 == 9:
                eta = self.get_eta(self.epoch, i)
                print(
                    f" epoch [{self.epoch:04d} / {self.args.epochs:04d}] [{i:04}/{steps}] eta: {eta:<20}  lr: \t{self.optimizer.param_groups[0]['lr']:0.4E} \tloss: {epoch_loss/i:0.5f}"
                )
        ret = self.computeAUROC(
            outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), "train"
        )
        self.epochs_stats["loss train"].append(epoch_loss / i)
        return ret

    def validate(self, dl):
        print(f"starting val epoch {self.epoch}")
        epoch_loss = 0
        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)

        with torch.no_grad():
            for i, (x, img, y_ehr, y_cxr, seq_lengths, pairs) in enumerate(dl):
                y = self.get_gt(y_ehr, y_cxr)
                x = torch.from_numpy(x).float()
                x = Variable(x.to(self.device), requires_grad=False)
                y = Variable(y.to(self.device), requires_grad=False)
                img = img.to(self.device)
                pairs = torch.FloatTensor(pairs).to(self.device)

                if (
                    self.args.task == "in-hospital-mortality"
                    or self.args.task == "readmission"
                ):
                    y = y.unsqueeze(1)
                out = self.model(
                    x, img, seq_lengths, pairs, self._get_alignment_lambda()
                )
                loss_ = self._compute_and_log_loss(out, y_gt=y, pairs=pairs, mode="val")
                if self.args.attn_fusion == "avg":
                    perd_final = (
                        out["pred_ehr"] + out["pred_cxr"] + out["pred_shared"]
                    ) / 3
                    pred_final = (1 - pairs.unsqueeze(1)) * (
                        out["pred_ehr"] + out["pred_shared"]
                    ) / 2 + pairs.unsqueeze(1) * perd_final
                else:
                    pred_final = out["pred_final"]
                loss = self._compute_masked_pred_loss(
                    pred_final, y, torch.ones_like(y[:, 0])
                )
                epoch_loss += loss.item()
                outPRED = torch.cat((outPRED, pred_final), 0)
                outGT = torch.cat((outGT, y), 0)

        self.scheduler.step(epoch_loss / len(self.val_dl))

        print(
            f"val [{self.epoch:04d} / {self.args.epochs:04d}] validation loss: \t{epoch_loss/i:0.5f}"
        )

        ret = self.computeAUROC(
            outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), "validation"
        )
        np.save(f"{self.args.save_dir}/pred.npy", outPRED.data.cpu().numpy())
        np.save(f"{self.args.save_dir}/gt.npy", outGT.data.cpu().numpy())

        self.epochs_stats["auroc val"].append(ret["auroc_mean"])
        self.epochs_stats["loss val"].append(epoch_loss / i)

        return ret
