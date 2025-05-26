CUDA_VISIBLE_DEVICES=2 CUDA_LAUNCH_BLOCKING=1 python fusion_main_mimic4.py \
--dim 256 --dropout 0 --layers 2 \
--vision_backbone resnet34 \
--mode train \
--epochs 100 --batch_size 32 --lr 8e-05 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs partial_ehr_cxr \
--data_ratio 1.0 \
--task in-hospital-mortality \
--labels_set mortality \
--fusion_type copula \
--copula_fuse_type lstm \
--copula 0.000001 \
--copula_resample False \
--save_dir checkpoints/mortality/mimic4/abalation/copula_partial_wo_resample.5
# nohup sh scripts/mortality/mimic4/abalation/copula_partial_wo_resample.sh > logs/ihm_mimic4/abalation/copula_partial_wo_resample.log 2>&1 &