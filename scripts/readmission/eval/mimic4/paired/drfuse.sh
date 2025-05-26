CUDA_VISIBLE_DEVICES=2 CUDA_LAUNCH_BLOCKING=1 python fusion_main_mimic4.py \
--dim 256 --dropout 0.3 --layers 2 \
--mode eval \
--epochs 100 --batch_size 32 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs paired_ehr_cxr \
--task readmission \
--fusion_type drfuse --layer_after 4 \
--labels_set readm \
--save_dir checkpoints/readm/mimic4/paired/drfuse
# nohup sh scripts/readmission/mimic4/paired/drfuse.sh > logs/readm_mimic4/paired/drfuse.log 2>&1 &