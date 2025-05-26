CUDA_VISIBLE_DEVICES=3 CUDA_LAUNCH_BLOCKING=1 python fusion_main_3d.py \
--dim 256 --dropout 0.3 --layers 2 \
--vision_backbone resnet34 \
--mode train \
--epochs 100 --batch_size 16 --lr 4.355e-05 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs paired_ehr_cxr \
--task readmission \
--fusion_type mmtm --layer_after 4 \
--labels_set readm \
--save_dir checkpoints/readm/mimic4/triple/mmtm
# nohup sh scripts/readmission/mimic4/triple/mmtm.sh > logs/readm_mimic4/triple/mmtm.log 2>&1 &