import argparse

def args_parser():
    parser = argparse.ArgumentParser(description='arguments')

    # task setting
    parser.add_argument('--task', type=str, default='readmission', help='train or eval for in-hospital-mortality, readmission, phenotyping')
    parser.add_argument('--labels_set', type=str, default='readm', help='pheno, mortality, readm, radiology')
    parser.add_argument('--num_classes', type=int, default=1, help='number of classes ihm:1, pheno:25')
    parser.add_argument('--vision_num_classes', default=1, type=int, help='number of classes ihm:1, pheno:25')

    # dataset setting
        # cxr setting
    parser.add_argument('--resize', default=256, type=int, help='number of epochs to train')
    parser.add_argument('--crop', default=224, type=int, help='number of epochs to train')
        # dataloader setting
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
        # data pairs setting 
    parser.add_argument('--data_pairs', type=str, default='paired_ehr_cxr', help='paired_ehr_cxr, partial_ehr_cxr, paired_ehr_note, partial_ehr_note')
    parser.add_argument('--data_ratio', type=float, default=1.0, help='percentage of uppaired data samples')
    parser.add_argument('--missing_token', type=str, default=None, help='zeros, learnable')
        # normalizer setting 
    parser.add_argument('--timestep', type=float, default=2.0, help="fixed timestep used in the dataset")
    parser.add_argument('--imputation', type=str, default='previous')
    parser.add_argument('--normalizer_state', type=str, default='data_mimic4/ihm_ts.normalizer',
                        help='Path to a state file of a normalizer. Leave none if you want to use one of the provided ones.')
    
    # training setting
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',  help='load imagenet pretrained model')
    parser.add_argument('--mode', type=str, default="train", help='mode: train or test')
    parser.add_argument('--eval', dest='eval', action='store_true',  help='eval the pretrained models on val and test split')
    parser.add_argument('--network', type=str)
    parser.add_argument('--epochs', type=int, default=50, help='number of chunks to train')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--beta_1', type=float, default=0.9, help='beta_1 param for Adam optimizer')
    parser.add_argument('--patience', type=int, default=15, help='number of epoch to wait for best')
    # Temperature annealing
    parser.add_argument('--temperature', type=float, default=0.0001)

    parser.add_argument('--load_state', type=str, default=None, help='state dir path')
    parser.add_argument('--load_state_cxr', type=str, default=None, help='state dir path')
    parser.add_argument('--load_state_ehr', type=str, default=None, help='state dir path')
    parser.add_argument('--load_state_note', type=str, default=None, help='state dir path')

    parser.add_argument('--resume', dest='resume', help='resume training from state to load', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--rec_dropout', type=float, default=0.0, help="dropout rate for recurrent connections")

    # model setting
    parser.add_argument('--fusion_type', type=str, default='copula', help='train or eval for [lstm, unified, mmtm, daft, copula, drfuse]')
        # backbone setting
        # vision backbone setting
    parser.add_argument('--layers', default=2, type=int, help='number of lstm stacked modules')
    parser.add_argument('--vision_backbone', default='resnet34', type=str, help='[densenet121, densenet169, densenet201]')
    parser.add_argument('--dim', type=int, default=256, help='number of hidden units')
    parser.add_argument('--depth', type=int, default=1, help='number of bi-LSTMs')
        # text backbone setting
    parser.add_argument('--bert_type', type=str, default="huawei-noah/TinyBERT_General_4L_312D", help='name of the bert pretrained model', 
                        choices=['huawei-noah/TinyBERT_General_4L_312D', 'emilyalsentzer/Bio_ClinicalBERT', 'allenai/biomed_roberta_base', 'bert-base-uncased', 'yikuan8/Clinical-Longformer'])
    parser.add_argument('--orig_d_txt', type=int, default=312, help='hidden_size of the bert model 768(max_length=1024)/312(max_length=512)')
    parser.add_argument('--d_txt', type=int, default=512, help='hidden_size of the bert model 768')
        # fusion setting 
    parser.add_argument('--fusion', type=str, default='joint', help='train or eval for [early late joint]')
    parser.add_argument('--align', type=float, default=0, help='align weight')
        # mmtm setting
    parser.add_argument('--layer_after', default=4, type=int, help='apply mmtm module after fourth layer -1 indicates mmtm after every layer')
    parser.add_argument('--mmtm_ratio', type=float, default=4, help='mmtm ratio hyperparameter')
        # daft setting
    parser.add_argument('--daft_activation', type=str, default='linear', help='daft activation ')
        # copula setting
    parser.add_argument('--copula', type=float, default=0.000001, help='copula weight')
    parser.add_argument('--copula_fuse_type', type=str, default=None, help='copula_fuse_type: lstm, mha')
    parser.add_argument('--copula_normalize_feats', action="store_true")
    parser.add_argument('--copula_resample', type=bool, default=True)
    parser.add_argument('--replace_w_align', type=str, default=None, help='replace_w_align: kl, cos')
        # Copula parameters
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--copula_family', type=str, default="Gumbel", choices=["Gumbel", "Clayton", "Frank", "Gaussian"])
    parser.add_argument('--rho_scale', type=float, default=-3)
        # drfuse setting   
    parser.add_argument('--lambda_disentangle_shared', type=float, default=1)
    parser.add_argument('--lambda_disentangle_ehr', type=float, default=1)
    parser.add_argument('--lambda_disentangle_cxr', type=float, default=1)
    parser.add_argument('--lambda_disentangle_note', type=float, default=1)
    parser.add_argument('--lambda_pred_ehr', type=float, default=1)
    parser.add_argument('--lambda_pred_cxr', type=float, default=1)
    parser.add_argument('--lambda_pred_note', type=float, default=1)
    parser.add_argument('--lambda_pred_shared', type=float, default=1)
    parser.add_argument('--aug_missing_ratio', type=float, default=0)
    parser.add_argument('--lambda_attn_aux', type=float, default=1)
    parser.add_argument('--ehr_n_layers', type=int, default=1)
    parser.add_argument('--ehr_n_head', type=int, default=4)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--adaptive_adc_lambda', action="store_true")
    parser.add_argument('--attn_fusion', action="store_true")
    parser.add_argument('--gamma', type=float, default=0)
    
    # path setting
    parser.add_argument('--ehr_data_dir', type=str, help='Path to the data of phenotyping fusion_type',
                        default='data_mimic4/')
    parser.add_argument('--cxr_data_dir', type=str, help='Path to the data of phenotyping fusion_type',
                        default='data_mimic4/mimic-cxr/')
    parser.add_argument('--save_dir', type=str, help='Directory relative which all output files are stored',
                    default='checkpoints/debug')

    # some default setting
    parser.set_defaults(copula_normalize_feats=True)
    return parser
