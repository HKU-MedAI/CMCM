CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 python fusion_main_mimic3.py \
--dim 256 --dropout 0 --layers 2 \
--vision_backbone resnet34 \
--mode eval \
--epochs 100 --batch_size 16 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs partial_ehr_note \
--data_ratio 1.0 \
--task in-hospital-mortality \
--labels_set mortality \
--fusion_type copula \
--copula_fuse_type lstm \
--copula 0.000001 \
--copula_family Gumbel \
--save_dir checkpoints/mortality/mimic3/partial/copula
# nohup sh scripts/mortality/mimic3/partial/copula.sh > logs/ihm_mimic3/partial/copula.log 2>&1 &