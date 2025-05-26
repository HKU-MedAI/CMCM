import math
import torch
from torch import nn
from torch.nn import functional as F
from models.ehr_transformer import EHRTransformer
from models.note_models import BertForRepresentation as NoteModel


class DrFuseModel(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_classes,
        ehr_dropout,
        ehr_n_layers,
        ehr_n_head,
        note_model="swin_s",
        logit_average=False,
        args=None,
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

        self.note_model_shared = NoteModel(args)
        self.note_model_shared_fc = nn.Linear(
            in_features=args.d_txt, out_features=hidden_size
        )
        self.note_model_spec = NoteModel(args)
        self.note_model_spec_fc = nn.Linear(
            in_features=args.d_txt, out_features=hidden_size
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
        self.note_model_linear = nn.Linear(
            in_features=hidden_size, out_features=num_classes
        )
        self.fuse_model_shared = nn.Linear(
            in_features=hidden_size, out_features=num_classes
        )

        # self.domain_classifier = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size//2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size//2, 1)
        # )
        self.attn_proj = nn.Linear(hidden_size, (2 + num_classes) * hidden_size)
        self.final_pred_fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, token, mask, seq_lengths, pairs, grl_lambda):
        feat_ehr_shared, feat_ehr_distinct, pred_ehr = self.ehr_model(x, seq_lengths)
        _, _, feat_note_shared = self.note_model_shared(token, mask)
        _, _, feat_note_distinct = self.note_model_spec(token, mask)

        feat_note_shared = self.note_model_shared_fc(feat_note_shared)
        feat_note_distinct = self.note_model_spec_fc(feat_note_distinct)

        # get shared feature
        pred_note = self.note_model_linear(feat_note_distinct).sigmoid()

        feat_ehr_shared = self.shared_project(feat_ehr_shared)
        feat_note_shared = self.shared_project(feat_note_shared)

        pairs = pairs.unsqueeze(1)

        h1 = feat_ehr_shared  # batch_size x hidden_size
        h2 = feat_note_shared
        term1 = torch.stack([h1 + h2, h1 + h2, h1, h2], dim=2)
        term2 = torch.stack([torch.zeros_like(h1), torch.zeros_like(h1), h1, h2], dim=2)
        feat_avg_shared = torch.logsumexp(term1, dim=2) - torch.logsumexp(term2, dim=2)

        feat_avg_shared = pairs * feat_avg_shared + (1 - pairs) * feat_ehr_shared
        pred_shared = self.fuse_model_shared(feat_avg_shared).sigmoid()

        # Disease-wise Attention
        attn_input = torch.stack(
            [feat_ehr_distinct, feat_avg_shared, feat_note_distinct], dim=1
        )
        qkvs = self.attn_proj(attn_input)
        q, v, *k = qkvs.chunk(2 + self.num_classes, dim=-1)

        # compute query vector
        q_mean = pairs * q.mean(dim=1) + (1 - pairs) * q[:, :-1].mean(dim=1)

        # compute attention weighting
        ks = torch.stack(k, dim=1)
        attn_logits = torch.einsum("bd,bnkd->bnk", q_mean, ks)
        attn_logits = attn_logits / math.sqrt(q.shape[-1])

        # filter out non-paired
        attn_mask = torch.ones_like(attn_logits)
        attn_mask[pairs.squeeze() == 0, :, -1] = 0
        attn_logits = attn_logits.masked_fill(attn_mask == 0, float("-inf"))
        attn_weights = F.softmax(attn_logits, dim=-1)

        # get final class-specific representation and prediction
        feat_final = torch.matmul(attn_weights, v)
        pred_final = self.final_pred_fc(feat_final)
        pred_final = torch.diagonal(pred_final, dim1=1, dim2=2).sigmoid()

        outputs = {
            "feat_ehr_shared": feat_ehr_shared,
            "feat_note_shared": feat_note_shared,
            "feat_ehr_distinct": feat_ehr_distinct,
            "feat_note_distinct": feat_note_distinct,
            "feat_final": feat_final,
            "pred_final": pred_final,
            "pred_shared": pred_shared,
            "pred_ehr": pred_ehr,
            "pred_note": pred_note,
            "attn_weights": attn_weights,
        }

        return outputs
