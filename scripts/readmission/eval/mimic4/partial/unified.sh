CUDA_VISIBLE_DEVICES=2 CUDA_LAUNCH_BLOCKING=1 python fusion_main_mimic4.py \
--dim 256 --dropout 0.3 --layers 2 \
--vision_backbone resnet34 \
--mode eval \
--epochs 100 --batch_size 32 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs partial_ehr_cxr \
--data_ratio 1.0 \
--task readmission \
--labels_set readm \
--fusion_type unified \
--save_dir checkpoints/readm/mimic4/partial/unified
# nohup sh scripts/readmission/mimic4/partial/unified.sh > logs/readm_mimic4/partial/unified.log 2>&1 &