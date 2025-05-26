CUDA_VISIBLE_DEVICES=2 CUDA_LAUNCH_BLOCKING=1 python fusion_main_mimic3.py \
--dim 256 --dropout 0.3 --layers 2 \
--vision_backbone resnet34 \
--mode eval \
--epochs 100 --batch_size 16 --lr 4.355e-05 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs partial_ehr_note \
--task in-hospital-mortality \
--fusion_type mmtm --layer_after 4 \
--labels_set mortality \
--save_dir checkpoints/mortality/mimic3/partial/mmtm

# nohup sh scripts/mortality/mimic3/partial/mmtm.sh > logs/ihm_mimic3/partial/mmtm.log 2>&1 &