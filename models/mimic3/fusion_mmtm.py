import torch.nn as nn
import torch
from models.loss import CosineLoss, KLDivLoss


class MMTM(nn.Module):
    def __init__(self, dim_txt, dim_ehr, ratio):
        super(MMTM, self).__init__()
        dim = dim_txt + dim_ehr
        dim_out = int(2 * dim / ratio)
        self.fc_squeeze = nn.Linear(dim, dim_out)

        self.fc_txt = nn.Linear(dim_out, dim_txt)
        self.fc_skeleton = nn.Linear(dim_out, dim_ehr)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, txt, skeleton):
        squeeze_array = []
        squeeze_array.append(txt)
        ehr_avg = torch.mean(skeleton, dim=1)
        squeeze_array.append(ehr_avg)

        squeeze = torch.cat(squeeze_array, 1)

        excitation = self.fc_squeeze(squeeze)
        excitation = self.relu(excitation)

        txt_out = self.fc_txt(excitation)
        sk_out = self.fc_skeleton(excitation)

        txt_out = self.sigmoid(txt_out)
        sk_out = self.sigmoid(sk_out)

        dim_diff = len(txt.shape) - len(txt_out.shape)
        txt_out = txt_out.view(txt_out.shape + (1,) * dim_diff)

        dim_diff = len(skeleton.shape) - len(sk_out.shape)
        sk_out = sk_out.view(sk_out.shape[0], 1, sk_out.shape[1])

        return txt * txt_out, skeleton * sk_out


class FusionMMTM(nn.Module):

    def __init__(self, args, ehr_model, note_model):

        super(FusionMMTM, self).__init__()
        self.args = args
        self.ehr_model = ehr_model
        self.note_model = note_model

        self.mmtm4 = MMTM(512, self.ehr_model.feats_dim, self.args.mmtm_ratio)

        feats_dim = 2 * self.note_model.d_txt

        self.joint_cls = nn.Sequential(
            nn.Linear(feats_dim, self.args.num_classes),
        )

        self.layer_after = args.layer_after
        self.projection = nn.Linear(self.ehr_model.feats_dim, self.note_model.d_txt)

        self.align_loss = CosineLoss()
        self.kl_loss = KLDivLoss()

    def forward(self, ehr, seq_lengths=None, token=None, mask=None):

        ehr = torch.nn.utils.rnn.pack_padded_sequence(
            ehr, seq_lengths, batch_first=True, enforce_sorted=False
        )

        ehr, (ht, _) = self.ehr_model.layer0(ehr)
        ehr_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(ehr, batch_first=True)

        _, _, note_feats = self.note_model(token, mask)

        if self.layer_after == 4 or self.layer_after == -1:
            note_feats, ehr_unpacked = self.mmtm4(note_feats, ehr_unpacked)

        note_preds = self.note_model.classifier(note_feats)
        note_preds_sig = torch.sigmoid(note_preds)

        ehr = torch.nn.utils.rnn.pack_padded_sequence(
            ehr_unpacked, seq_lengths, batch_first=True, enforce_sorted=False
        )
        ehr, (ht, _) = self.ehr_model.layer1(ehr)
        ehr_feats = ht.squeeze(0)

        ehr_feats = self.ehr_model.do(ehr_feats)
        ehr_preds = self.ehr_model.dense_layer(ehr_feats)
        ehr_preds_sig = torch.sigmoid(ehr_preds)

        late_average = (note_preds + ehr_preds) / 2
        late_average_sig = (note_preds_sig + ehr_preds_sig) / 2

        projected = self.projection(ehr_feats)
        loss = self.kl_loss(note_feats, projected)

        feats = torch.cat([projected, note_feats], dim=1)
        joint_preds = self.joint_cls(feats)

        joint_preds_sig = torch.sigmoid(joint_preds)

        return {
            "note_only": note_preds_sig,
            "ehr_only": ehr_preds_sig,
            "joint": joint_preds_sig,
            "late_average": late_average_sig,
            "align_loss": loss,
            "note_only_scores": note_preds,
            "ehr_only_scores": ehr_preds,
            "late_average_scores": late_average,
            "joint_scores": joint_preds,
        }
