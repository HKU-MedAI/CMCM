import torch
from torch import nn
from transformers import AutoModel


class BertForRepresentation(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bert_type = args.bert_type
        self.orig_d_txt = args.orig_d_txt  # hidden_size of the bert model 768/312
        self.d_txt = args.d_txt
        self.bert = AutoModel.from_pretrained(self.bert_type)
        self.dropout = torch.nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.proj_txt = nn.Linear(self.orig_d_txt, self.d_txt)
        self.bce_loss = torch.nn.BCELoss(size_average=True)
        self.classifier = nn.Sequential(nn.Linear(self.d_txt, args.num_classes))
        self.feats_dim = self.d_txt
        num_params = sum(p.numel() for p in self.bert.parameters())
        print(f"bert model size: {num_params}")

    def forward(self, input_ids_sequence, attention_mask_sequence, labels=None):
        txt_arr = []
        for input_ids, attention_mask in zip(
            input_ids_sequence, attention_mask_sequence
        ):
            if "Longformer" in self.bert_type:
                global_attention_mask = torch.clamp(attention_mask.clone() - 1, 0, 1)
                attention_mask = torch.clamp(attention_mask, 0, 1)
                text_embeddings = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    global_attention_mask=global_attention_mask,
                )
            else:
                text_embeddings = self.bert(input_ids, attention_mask=attention_mask)
            text_embeddings = text_embeddings[0][:, 0, :]
            text_embeddings = self.dropout(text_embeddings)
            txt_arr.append(text_embeddings)
        txt_arr = torch.stack(txt_arr)  # bs * num_notes * hidden_size(768/312)

        # maybe use other aggregation methods
        x_txt = torch.mean(txt_arr, dim=1)  # bs * hidden_size(768/312)
        txt_feats = self.proj_txt(x_txt)  # bs * d_txt

        preds = self.classifier(txt_feats)
        preds = torch.sigmoid(preds)
        lossvalue_bce = torch.zeros(1).to(input_ids_sequence.device)

        if labels is not None:
            lossvalue_bce = self.bce_loss(preds, labels)
        return preds, lossvalue_bce, txt_feats


if __name__ == "__main__":
    import sys

    sys.path.append("/home/fwu/Documents/myProjects/MedFuse/")
    from torch.utils.data import DataLoader
    from arguments import args_parser
    from dataset_mimic3.note_dataset import get_note_datasets

    parser = args_parser()
    args = parser.parse_args()
    args.ehr_data_dir = "/disk1/fwu/myProjects/MedFuse/data_mimic3"
    args.labels_set = "ihm"
    args.task = "in-hospital-mortality"
    args.bert_type = "huawei-noah/TinyBERT_General_4L_312D"

    dataset_train, dataset_validate, dataset_test = get_note_datasets(args)

    val_loader = DataLoader(dataset_validate, batch_size=4, shuffle=False)
    batch = {}
    for batch in val_loader:
        break
    model = BertForRepresentation(args)
    text_token, atten_mask, label = batch
    preds, lossvalue_bce, txt_feats = model(text_token, atten_mask)
    print(txt_feats.shape)
    """
    torch.Size([4, 512])
    """
