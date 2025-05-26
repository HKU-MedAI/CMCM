CUDA_VISIBLE_DEVICES=2 CUDA_LAUNCH_BLOCKING=1 python fusion_main_mimic4.py \
--dim 256 --dropout 0.3 --layers 2 \
--vision_backbone resnet34 \
--mode train \
--epochs 100 --batch_size 32 --lr 6.493e-05 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs paired_ehr_cxr \
--fusion_type daft --layer_after 4 \
--labels_set mortality \
--task in-hospital-mortality \
--save_dir checkpoints/mortality/mimic4/paired/daft

# nohup sh scripts/mortality/mimic4/paired/daft.sh > logs/ihm_mimic4/paired/daft.log 2>&1 &