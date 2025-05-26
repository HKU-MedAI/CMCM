CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python fusion_main_mimic3.py \
--dim 256 --dropout 0.3 --layers 2 \
--mode eval \
--epochs 100 --batch_size 8 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs paired_ehr_note \
--task readmission \
--fusion_type drfuse --layer_after 4 \
--labels_set readm \
--save_dir checkpoints/readm/mimic3/paired/drfuse 

# nohup sh scripts/readmission/mimic3/paired/drfuse.sh > logs/readm_mimic3/paired/drfuse.log 2>&1 &