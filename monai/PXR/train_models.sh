#!/bin/bash

MODELS=('densenet121' 'densenet169' 'densenet201' 'densenet264' 'senet154' 'se_resnet50' 'se_resnet101' 'se_resnet152' 'se_resnext50_32x4d' 'se_resnext101_32x4d')

for m in "${MODELS[@]}"
do
    echo $m
    python train.py \
        -d /data/no_mask_data_list.json \
        -r /data/preprocessing \
        -e 200 \
        -b 20 \
        -m $m -c models/$m.pth
done

