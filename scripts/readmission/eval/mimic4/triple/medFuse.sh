CUDA_VISIBLE_DEVICES=5 CUDA_LAUNCH_BLOCKING=1 python fusion_main_3d.py \
--dim 256 --dropout 0.3 --layers 2 \
--lr 9.382e-05 \
--vision_backbone resnet34 \
--copula 0 \
--mode eval \
--epochs 100 --batch_size 16 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs paired_ehr_cxr \
--data_ratio 1.0 \
--task readmission \
--labels_set readm \
--fusion_type lstm \
--save_dir checkpoints/readm/mimic4/triple/medFuse
# nohup sh scripts/readmission/mimic4/triple/medFuse.sh > logs/readm_mimic4/triple/medFuse.log 2>&1 &