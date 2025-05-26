CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python fusion_main_3d.py \
--dim 256 --dropout 0.3 --layers 2 \
--vision_backbone resnet34 \
--mode train \
--epochs 100 --batch_size 16 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs paired_ehr_cxr \
--data_ratio 1.0 \
--task readmission \
--labels_set readm \
--fusion_type unified \
--save_dir checkpoints/readm/mimic4/triple/unified
# nohup sh scripts/readmission/mimic4/triple/unified.sh > logs/readm_mimic4/triple/unified.log 2>&1 &