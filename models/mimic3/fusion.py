import torch.nn as nn
import torch
import numpy as np
from models.loss import CosineLoss, KLDivLoss, CopulaLoss


class Fusion(nn.Module):
    def __init__(self, args, ehr_model, note_model):

        super(Fusion, self).__init__()
        self.args = args
        self.ehr_model = ehr_model
        self.note_model = note_model

        target_classes = self.args.num_classes
        lstm_in = self.ehr_model.feats_dim
        lstm_out = self.note_model.feats_dim
        projection_in = self.note_model.feats_dim

        self.projection = nn.Linear(projection_in, lstm_in)
        feats_dim = 2 * self.ehr_model.feats_dim

        self.fused_cls = nn.Sequential(
            nn.Linear(feats_dim, self.args.num_classes), nn.Sigmoid()
        )

        self.align_loss = CosineLoss()
        self.kl_loss = KLDivLoss()
        self.copula_loss = CopulaLoss()

        self.lstm_fused_cls = nn.Sequential(
            nn.Linear(lstm_out, target_classes), nn.Sigmoid()
        )

        self.lstm_fusion_layer = nn.LSTM(
            lstm_in, lstm_out, batch_first=True, dropout=0.0
        )

    def forward_uni_note(self, x, seq_lengths=None, token=None, mask=None):
        note_preds, _, feats = self.note_model(token, mask)
        return {"uni_note": note_preds, "note_feats": feats}

    def forward(self, x, seq_lengths=None, token=None, mask=None, pairs=None):
        if self.args.fusion_type == "uni_note":
            return self.forward_uni_note(
                x, seq_lengths=seq_lengths, token=token, mask=mask
            )
        elif self.args.fusion_type in ["joint", "early", "late_avg", "unified"]:
            return self.forward_fused(
                x, seq_lengths=seq_lengths, token=token, mask=mask, pairs=pairs
            )
        elif self.args.fusion_type == "uni_ehr":
            return self.forward_uni_ehr(
                x, seq_lengths=seq_lengths, token=token, mask=mask
            )
        elif self.args.fusion_type == "lstm":
            return self.forward_lstm_fused(
                x, seq_lengths=seq_lengths, token=token, mask=mask, pairs=pairs
            )
        elif self.args.fusion_type == "uni_ehr_lstm":
            return self.forward_lstm_ehr(
                x, seq_lengths=seq_lengths, token=token, mask=mask, pairs=pairs
            )

    def forward_uni_ehr(self, x, seq_lengths=None, token=None, mask=None):
        ehr_preds, feats = self.ehr_model(x, seq_lengths)
        return {"uni_ehr": ehr_preds, "ehr_feats": feats}

    def forward_fused(self, x, seq_lengths=None, token=None, mask=None, pairs=None):

        ehr_preds, ehr_feats = self.ehr_model(x, seq_lengths)
        note_preds, _, note_feats = self.note_model(token, mask)
        projected = self.projection(note_feats)

        loss = self.align_loss(projected, ehr_feats)

        feats = torch.cat([ehr_feats, projected], dim=1)
        fused_preds = self.fused_cls(feats)

        late_avg = (note_preds + ehr_preds) / 2
        return {
            "early": fused_preds,
            "joint": fused_preds,
            "late_avg": late_avg,
            "align_loss": loss,
            "ehr_feats": ehr_feats,
            "note_feats": projected,
            "unified": fused_preds,
        }

    def forward_lstm_fused(
        self, x, seq_lengths=None, token=None, mask=None, pairs=None
    ):
        if self.args.labels_set == "note":
            _, ehr_feats = self.ehr_model(x, seq_lengths)

            _, _, note_feats = self.note_model(token, mask)

            feats = note_feats[:, None, :]

            ehr_feats = self.projection(ehr_feats)

            ehr_feats[list(~np.array(pairs))] = 0
            feats = torch.cat([feats, ehr_feats[:, None, :]], dim=1)
        else:

            _, ehr_feats = self.ehr_model(x, seq_lengths)
            _, _, note_feats = self.note_model(token, mask)
            note_feats = self.projection(note_feats)

            note_feats[list(~np.array(pairs))] = 0
            if len(ehr_feats.shape) == 1:
                feats = ehr_feats[None, None, :]
                feats = torch.cat([feats, note_feats[:, None, :]], dim=1)
            else:
                feats = ehr_feats[:, None, :]
                feats = torch.cat([feats, note_feats[:, None, :]], dim=1)
        seq_lengths = np.array([1] * len(seq_lengths))
        seq_lengths[pairs] = 2

        feats = torch.nn.utils.rnn.pack_padded_sequence(
            feats, seq_lengths, batch_first=True, enforce_sorted=False
        )

        x, (ht, _) = self.lstm_fusion_layer(feats)

        out = ht.squeeze(0)

        fused_preds = self.lstm_fused_cls(out)

        return {
            "lstm": fused_preds,
            "ehr_feats": ehr_feats,
            "note_feats": note_feats,
        }

    def forward_lstm_ehr(self, x, seq_lengths=None, token=None, mask=None, pairs=None):
        _, ehr_feats = self.ehr_model(x, seq_lengths)
        feats = ehr_feats[:, None, :]

        seq_lengths = np.array([1] * len(seq_lengths))

        feats = torch.nn.utils.rnn.pack_padded_sequence(
            feats, seq_lengths, batch_first=True, enforce_sorted=False
        )

        x, (ht, _) = self.lstm_fusion_layer(feats)

        out = ht.squeeze(0)

        fused_preds = self.lstm_fused_cls(out)

        return {
            "uni_ehr_lstm": fused_preds,
        }
