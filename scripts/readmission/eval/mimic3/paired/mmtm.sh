CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 python fusion_main_mimic3.py \
--dim 256 --dropout 0.3 --layers 2 \
--vision_backbone resnet34 \
--mode eval \
--epochs 100 --batch_size 16 --lr 4.355e-05 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs paired_ehr_note \
--task readmission \
--fusion_type mmtm --layer_after 4 \
--labels_set readm \
--save_dir checkpoints/readm/mimic3/paired/mmtm
# nohup sh scripts/readmission/mimic3/paired/mmtm.sh > logs/readm_mimic3/paired/mmtm.log 2>&1 &