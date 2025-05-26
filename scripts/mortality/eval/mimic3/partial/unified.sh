CUDA_VISIBLE_DEVICES=2 CUDA_LAUNCH_BLOCKING=1 python fusion_main_mimic3.py \
--dim 256 --dropout 0.3 --layers 2 \
--vision_backbone resnet34 \
--mode eval \
--epochs 100 --batch_size 16 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs partial_ehr_note \
--data_ratio 1.0 \
--task in-hospital-mortality \
--labels_set mortality \
--fusion_type unified \
--save_dir checkpoints/mortality/mimic3/partial/unified
# nohup sh scripts/mortality/mimic3/partial/unified.sh > logs/ihm_mimic3/partial/unified.log 2>&1 &