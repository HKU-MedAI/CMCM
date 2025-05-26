CUDA_VISIBLE_DEVICES=2 CUDA_LAUNCH_BLOCKING=1 python fusion_benchmark_mimic3.py \
--dim 256 --dropout 0 --layers 2 \
--mode eval \
--epochs 100 --batch_size 16 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs paired_ehr_note \
--data_ratio 1.0 \
--task readmission \
--labels_set readm \
--fusion_type copula \
--copula_fuse_type lstm \
--copula 0.000001 \
--copula_family Gumbel \
--save_dir checkpoints/readm/mimic3/paired/copula
# nohup sh scripts/readmission/mimic3/paired/copula.sh > logs/readm_mimic4/paired/copula.log 2>&1 &