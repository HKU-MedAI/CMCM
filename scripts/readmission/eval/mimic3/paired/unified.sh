CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python fusion_main_mimic3.py \
--dim 256 --dropout 0.3 --layers 2 \
--vision_backbone resnet34 \
--mode eval \
--epochs 100 --batch_size 16 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs paired_ehr_note \
--data_ratio 1.0 \
--task readmission \
--labels_set readm \
--fusion_type unified \
--save_dir checkpoints/readm/mimic3/paired/unified
# nohup sh scripts/readmission/mimic3/paired/unified.sh > logs/readm_mimic3/paired/unified.log 2>&1 &