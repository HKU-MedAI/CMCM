CUDA_VISIBLE_DEVICES=2 CUDA_LAUNCH_BLOCKING=1 python fusion_main_mimic4.py \
--dim 256 --dropout 0 --layers 2 \
--vision_backbone resnet34 \
--mode train \
--epochs 100 --batch_size 32 --lr 1e-04 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs paired_ehr_cxr \
--data_ratio 1.0 \
--task in-hospital-mortality \
--labels_set mortality \
--fusion_type copula \
--copula_fuse_type lstm \
--replace_w_align kl \
--copula 1 \
--save_dir checkpoints/mortality/abalation/copula_paired_w_align
# nohup sh scripts/mortality/train/abalation/copula_paired_w_align.sh > logs/ihm_mimic4/abalation/copula_paired_w_align.log 2>&1 &