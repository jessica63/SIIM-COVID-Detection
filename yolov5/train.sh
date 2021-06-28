#!/bin/bash

for FOLD in $(seq 1 4)
do

    CUDA_VISIBLE_DEVICES=2 python train.py \
        --img 1024 \
        --batch 10 \
        --epochs 25 \
        --hyp ./data/hyp.customized.yaml \
        --data ./data/opacity_${FOLD}.yaml \
        --weights yolov5x.pt \
        --cache \
        --project covid \
        --name 0623_fold_${FOLD} \
        --entity jessica63
done
