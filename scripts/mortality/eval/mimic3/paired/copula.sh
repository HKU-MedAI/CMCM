CUDA_VISIBLE_DEVICES=2 CUDA_LAUNCH_BLOCKING=1 python fusion_main_mimic3.py \
--dim 256 --dropout 0 --layers 2 \
--mode eval \
--lr 0.0001 \
--epochs 100 --batch_size 16 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs paired_ehr_note \
--data_ratio 1.0 \
--task in-hospital-mortality \
--labels_set mortality \
--fusion_type copula \
--copula_fuse_type lstm \
--copula 0.00001 \
--copula_family Frank \
--save_dir checkpoints/mortality/mimic3/paired/copula
# nohup sh scripts/mortality/mimic3/paired/copula.sh > logs/ihm_mimic3/paired/copula.log 2>&1 &