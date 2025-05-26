import torch.nn as nn
import torch
import numpy as np
from models.loss import CosineLoss, KLDivLoss, CopulaLoss
import torch.nn.functional as F


class Copula_Fusion(nn.Module):
    def __init__(self, args, ehr_model, cxr_model):

        super(Copula_Fusion, self).__init__()
        self.args = args
        self.ehr_model = ehr_model
        self.cxr_model = cxr_model

        target_classes = self.args.num_classes
        lstm_in = self.ehr_model.feats_dim
        lstm_out = self.cxr_model.feats_dim
        projection_in = self.cxr_model.feats_dim

        self.projection = nn.Linear(projection_in, lstm_in)
        feats_dim = 2 * self.ehr_model.feats_dim

        self.fused_cls = nn.Sequential(
            nn.Linear(feats_dim, self.args.num_classes), nn.Sigmoid()
        )

        self.align_loss = CosineLoss()
        self.kl_loss = KLDivLoss()
        self.copula_loss = CopulaLoss(
            K=args.K, rho_scale=args.rho_scale, family=args.copula_family
        )

        self.lstm_fused_cls = nn.Sequential(
            nn.Linear(lstm_out, target_classes), nn.Sigmoid()
        )

        self.lstm_fusion_layer = nn.LSTM(
            lstm_in, lstm_out, batch_first=True, dropout=0.0
        )

    def forward(self, x, seq_lengths=None, img=None, pairs=None):

        ehr_preds, ehr_feats = self.ehr_model(x, seq_lengths)
        cxr_preds, _, cxr_feats = self.cxr_model(img)
        projected = self.projection(cxr_feats)

        # normalize the ehr_feats&cxr_feats
        if self.args.copula_normalize_feats:
            ehr_feats = F.normalize(ehr_feats, p=2, dim=1)
            projected = F.normalize(projected, p=2, dim=1)

        if self.args.copula_resample:
            n_samples = len(projected[list(~np.array(pairs))])
            if n_samples > 0:
                cxr_samples = self.copula_loss.rsample(
                    n_samples=torch.zeros(n_samples).size()
                )
                projected[list(~np.array(pairs))] = cxr_samples.detach()
        else:
            projected[list(~np.array(pairs))] = 0

        if self.args.replace_w_align == "kl":
            copula_loss = self.kl_loss(ehr_feats, projected)
        elif self.args.replace_w_align == "cos":
            copula_loss = self.align_loss(ehr_feats, projected)
        else:
            copula_loss = self.copula_loss(ehr_feats, projected)

        if self.args.copula_fuse_type == "lstm":
            if len(ehr_feats.shape) == 1:
                feats = ehr_feats[None, None, :]
                feats = torch.cat([feats, projected[:, None, :]], dim=1)
            else:
                feats = ehr_feats[:, None, :]
                feats = torch.cat([feats, projected[:, None, :]], dim=1)
            seq_lengths = np.array([1] * len(seq_lengths))
            seq_lengths[pairs] = 2

            feats = torch.nn.utils.rnn.pack_padded_sequence(
                feats, seq_lengths, batch_first=True, enforce_sorted=False
            )

            x, (ht, _) = self.lstm_fusion_layer(feats)

            out = ht.squeeze(0)

            fused_preds = self.lstm_fused_cls(out)
        else:
            feats = torch.cat([ehr_feats, projected], dim=1)
            fused_preds = self.fused_cls(feats)

        return {
            "ehr_feats": ehr_feats,
            "cxr_feats": projected,
            "copula": fused_preds,
            "copula_loss": copula_loss,
        }
