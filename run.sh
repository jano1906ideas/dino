#!/bin/bash


python main_dino.py --arch vit_small \
    --batch_size_per_gpu 16 \
    --patch_size 8 \
    --data_path /home/jano1906/datasets/imagenet/ILSVRC/Data/CLS-LOC \
    --output_dir logs \
    --norm_last_layer False \
    --finetune /home/jano1906/git/dino/checkpoints/dino_deitsmall8_pretrain_full_checkpoint.pth \
    --sample_divisions \
    --saveckp_freq 5 \
    --epochs 50 \
    --lr 1e-6 \
    --local_crops_number 10