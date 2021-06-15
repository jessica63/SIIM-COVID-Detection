#!/bin/bash

MODELS=('densenet121' 'densenet169' 'densenet201' 'densenet264' 'senet154' 'se_resnet50' 'se_resnet101' 'se_resnet152' 'se_resnext50_32x4d' 'se_resnext101_32x4d')

mkdir -p results
for m in "${MODELS[@]}"
do
    python evaluate.py -d /data/no_mask_data_list.json -r /data/preprocessing -k training   -c models/$m.pth -m $m >  results/$m.txt
    python evaluate.py -d /data/no_mask_data_list.json -r /data/preprocessing -k validation -c models/$m.pth -m $m >> results/$m.txt
    python evaluate.py -d /data/no_mask_data_list.json -r /data/preprocessing -k test       -c models/$m.pth -m $m >> results/$m.txt
done

python summarize.py
