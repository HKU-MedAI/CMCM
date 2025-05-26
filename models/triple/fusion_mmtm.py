import torch.nn as nn
import torch
from models.loss import CosineLoss, KLDivLoss


class MMTM_V(nn.Module):
    def __init__(self, dim_visual, dim_ehr, ratio):
        super(MMTM_V, self).__init__()
        dim = dim_visual + dim_ehr
        dim_out = int(2 * dim / ratio)
        self.fc_squeeze = nn.Linear(dim, dim_out)

        self.fc_visual = nn.Linear(dim_out, dim_visual)
        self.fc_skeleton = nn.Linear(dim_out, dim_ehr)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, visual, skeleton):
        squeeze_array = []
        visual_view = visual.view(
            visual.shape[:2] + (-1,)
        )  # bs, 512, 7, 7 -> bs, 512, 49
        squeeze_array.append(torch.mean(visual_view, dim=-1))  # bs, 512, 49 -> bs, 512
        ehr_avg = torch.mean(skeleton, dim=1)  # bs, seq_len, 256 -> bs, 256

        squeeze_array.append(ehr_avg)

        squeeze = torch.cat(squeeze_array, 1)

        excitation = self.fc_squeeze(squeeze)  # bs, 512+256=768 -> bs, 768/ratio
        excitation = self.relu(excitation)

        vis_out = self.fc_visual(excitation)
        sk_out = self.fc_skeleton(excitation)

        vis_out = self.sigmoid(vis_out)  # bs, 512
        sk_out = self.sigmoid(sk_out)  # bs, 256

        dim_diff = len(visual.shape) - len(vis_out.shape)
        vis_out = vis_out.view(
            vis_out.shape + (1,) * dim_diff
        )  # bs, 512 -> bs, 512, 1, 1

        dim_diff = len(skeleton.shape) - len(sk_out.shape)
        sk_out = sk_out.view(
            sk_out.shape[0], 1, sk_out.shape[1]
        )  # bs, 256 -> bs, 1, 256

        return visual * vis_out, skeleton * sk_out


class MMTM_N(nn.Module):
    def __init__(self, dim_txt, dim_ehr, ratio):
        super(MMTM_N, self).__init__()
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
    def __init__(self, args, ehr_model, cxr_model, note_model):
        super(FusionMMTM, self).__init__()
        self.args = args
        self.ehr_model = ehr_model
        self.cxr_model = cxr_model
        self.note_model = note_model

        self.mmtm4_v = MMTM_V(512, self.ehr_model.feats_dim, self.args.mmtm_ratio)
        self.mmtm4_n = MMTM_N(512, self.ehr_model.feats_dim, self.args.mmtm_ratio)

        feats_dim = 3 * self.cxr_model.feats_dim

        self.joint_cls = nn.Sequential(
            nn.Linear(feats_dim, self.args.num_classes),
        )

        self.layer_after = args.layer_after
        self.projection_ehr = nn.Linear(
            self.ehr_model.feats_dim, self.cxr_model.feats_dim
        )
        self.align_loss = CosineLoss()
        self.kl_loss = KLDivLoss()

    def forward(self, ehr, seq_lengths=None, img=None, token=None, mask=None):
        ehr = torch.nn.utils.rnn.pack_padded_sequence(
            ehr, seq_lengths, batch_first=True, enforce_sorted=False
        )

        ehr, (ht, _) = self.ehr_model.layer0(ehr)
        ehr_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(
            ehr, batch_first=True
        )  # bs, seq_len, feats_dim(256)

        _, _, note_feats = self.note_model(token, mask)  # bs, 512

        cxr_feats = self.cxr_model.vision_backbone.conv1(img)
        cxr_feats = self.cxr_model.vision_backbone.bn1(cxr_feats)
        cxr_feats = self.cxr_model.vision_backbone.relu(cxr_feats)
        cxr_feats = self.cxr_model.vision_backbone.maxpool(cxr_feats)  # bs, 64, 56, 56
        cxr_feats = self.cxr_model.vision_backbone.layer1(cxr_feats)  # bs, 64, 56, 56
        cxr_feats = self.cxr_model.vision_backbone.layer2(cxr_feats)  # bs, 128, 28, 28
        cxr_feats = self.cxr_model.vision_backbone.layer3(cxr_feats)  # bs, 256, 14, 14
        cxr_feats = self.cxr_model.vision_backbone.layer4(cxr_feats)  # bs, 512, 7, 7

        # 512
        if self.layer_after == 4 or self.layer_after == -1:
            cxr_feats, ehr_unpacked = self.mmtm4_v(cxr_feats, ehr_unpacked)
            note_feats, ehr_unpacked = self.mmtm4_v(note_feats, ehr_unpacked)

        cxr_feats = self.cxr_model.vision_backbone.avgpool(cxr_feats)  # bs, 512, 1, 1
        cxr_feats = torch.flatten(cxr_feats, 1)  # bs, 512

        cxr_preds = self.cxr_model.classifier(cxr_feats)
        cxr_preds_sig = torch.sigmoid(cxr_preds)

        note_preds = self.note_model.classifier(note_feats)
        note_preds_sig = torch.sigmoid(note_preds)

        ehr = torch.nn.utils.rnn.pack_padded_sequence(
            ehr_unpacked, seq_lengths, batch_first=True, enforce_sorted=False
        )
        ehr, (ht, _) = self.ehr_model.layer1(ehr)
        ehr_feats = ht.squeeze(0)  # bs, 256

        ehr_feats = self.ehr_model.do(ehr_feats)
        ehr_preds = self.ehr_model.dense_layer(ehr_feats)
        ehr_preds_sig = torch.sigmoid(ehr_preds)

        late_average = (cxr_preds + ehr_preds + note_preds) / 3
        late_average_sig = (cxr_preds_sig + ehr_preds_sig + note_preds_sig) / 3

        projected_ehr = self.projection_ehr(ehr_feats)  # bs, 512
        loss = self.kl_loss(cxr_feats, projected_ehr) + self.kl_loss(
            note_feats, projected_ehr
        )

        feats = torch.cat([projected_ehr, cxr_feats, note_feats], dim=1)  # bs, 512*3
        joint_preds = self.joint_cls(feats)

        joint_preds_sig = torch.sigmoid(joint_preds)

        return {
            "cxr_only": cxr_preds_sig,
            "ehr_only": ehr_preds_sig,
            "note_only": note_preds_sig,
            "joint": joint_preds_sig,
            "late_average": late_average_sig,
            "align_loss": loss,
            "cxr_only_scores": cxr_preds,
            "ehr_only_scores": ehr_preds,
            "note_only_scores": note_preds,
            "late_average_scores": late_average,
            "joint_scores": joint_preds,
        }
