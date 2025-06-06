import math
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50
from models.ehr_transformer import EHRTransformer


class DrFuseModel(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_classes,
        ehr_dropout,
        ehr_n_layers,
        ehr_n_head,
        cxr_model="swin_s",
        logit_average=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.logit_average = logit_average
        self.ehr_model = EHRTransformer(
            input_size=76,
            num_classes=num_classes,
            d_model=hidden_size,
            n_head=ehr_n_head,
            n_layers_feat=1,
            n_layers_shared=ehr_n_layers,
            n_layers_distinct=ehr_n_layers,
            dropout=ehr_dropout,
        )

        resnet = resnet50()
        # should use all stage0 of resnet50
        self.cxr_model_feat = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,  # missing in drfuse
            resnet.maxpool,  # missing in drfuse
        )

        resnet = resnet50()
        self.cxr_model_shared = nn.Sequential(
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,  # missing in drfuse
        )
        self.cxr_model_shared_fc = nn.Linear(
            in_features=resnet.fc.in_features, out_features=hidden_size
        )

        resnet = resnet50()
        self.cxr_model_spec = nn.Sequential(
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,  # missing in drfuse
        )
        self.cxr_model_spec_fc = nn.Linear(
            in_features=resnet.fc.in_features, out_features=hidden_size
        )

        self.shared_project = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.ehr_model_linear = nn.Linear(
            in_features=hidden_size, out_features=num_classes
        )
        self.cxr_model_linear = nn.Linear(
            in_features=hidden_size, out_features=num_classes
        )
        self.fuse_model_shared = nn.Linear(
            in_features=hidden_size, out_features=num_classes
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.attn_proj = nn.Linear(hidden_size, (2 + num_classes) * hidden_size)
        self.final_pred_fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, img, seq_lengths, pairs, grl_lambda):
        feat_ehr_shared, feat_ehr_distinct, pred_ehr = self.ehr_model(
            x, seq_lengths
        )  # batch_size x hidden_size(256)
        feat_cxr = self.cxr_model_feat(img)
        feat_cxr_shared = self.cxr_model_shared(feat_cxr)  # batch_size x 64 x 56 x 56
        feat_cxr_distinct = self.cxr_model_spec(feat_cxr)
        # fix the bug in drfuse
        feat_cxr_shared = feat_cxr_shared.view(feat_cxr_shared.size(0), -1)
        feat_cxr_distinct = feat_cxr_distinct.view(feat_cxr_distinct.size(0), -1)
        feat_cxr_shared = self.cxr_model_shared_fc(
            feat_cxr_shared
        )  # batch_size x hidden_size(256)
        feat_cxr_distinct = self.cxr_model_spec_fc(
            feat_cxr_distinct
        )  # batch_size x hidden_size(256)

        # get shared feature
        pred_cxr = self.cxr_model_linear(feat_cxr_distinct).sigmoid()

        feat_ehr_shared = self.shared_project(
            feat_ehr_shared
        )  # batch_size x hidden_size(256)
        feat_cxr_shared = self.shared_project(
            feat_cxr_shared
        )  # batch_size x hidden_size(256)

        pairs = pairs.unsqueeze(1)

        h1 = feat_ehr_shared  # batch_size x hidden_size
        h2 = feat_cxr_shared
        term1 = torch.stack(
            [h1 + h2, h1 + h2, h1, h2], dim=2
        )  # batch_size x hidden_size x 4
        term2 = torch.stack(
            [torch.zeros_like(h1), torch.zeros_like(h1), h1, h2], dim=2
        )  # batch_size x hidden_size x 4
        feat_avg_shared = torch.logsumexp(term1, dim=2) - torch.logsumexp(
            term2, dim=2
        )  # batch_size x hidden_size

        feat_avg_shared = pairs * feat_avg_shared + (1 - pairs) * feat_ehr_shared
        pred_shared = self.fuse_model_shared(feat_avg_shared).sigmoid()

        # Disease-wise Attention
        attn_input = torch.stack(
            [feat_ehr_distinct, feat_avg_shared, feat_cxr_distinct], dim=1
        )  # batch_size x 3 x hidden_size(256)
        qkvs = self.attn_proj(
            attn_input
        )  # batch_size x (2+num_classes)*hidden_size(256)
        q, v, *k = qkvs.chunk(2 + self.num_classes, dim=-1)

        # compute query vector
        q_mean = pairs * q.mean(dim=1) + (1 - pairs) * q[:, :-1].mean(
            dim=1
        )  # batch_size x hidden_size(256)

        # compute attention weighting
        ks = torch.stack(k, dim=1)  # batch_size x 1 x 2+num_classes x hidden_size(256)
        attn_logits = torch.einsum(
            "bd,bnkd->bnk", q_mean, ks
        )  # batch_size x 1 x 2+num_classes
        attn_logits = attn_logits / math.sqrt(
            q.shape[-1]
        )  # batch_size x 1 x 2+num_classes

        # filter out non-paired
        attn_mask = torch.ones_like(attn_logits)
        attn_mask[pairs.squeeze() == 0, :, -1] = 0
        attn_logits = attn_logits.masked_fill(attn_mask == 0, float("-inf"))
        attn_weights = F.softmax(attn_logits, dim=-1)

        # get final class-specific representation and prediction
        feat_final = torch.matmul(attn_weights, v)  # batch_size x 1 x hidden_size(256)
        pred_final = self.final_pred_fc(feat_final)
        pred_final = torch.diagonal(pred_final, dim1=1, dim2=2).sigmoid()

        outputs = {
            "feat_ehr_shared": feat_ehr_shared,
            "feat_cxr_shared": feat_cxr_shared,
            "feat_ehr_distinct": feat_ehr_distinct,
            "feat_cxr_distinct": feat_cxr_distinct,
            "feat_final": feat_final,
            "pred_final": pred_final,
            "pred_shared": pred_shared,
            "pred_ehr": pred_ehr,
            "pred_cxr": pred_cxr,
            "attn_weights": attn_weights,
        }

        return outputs
