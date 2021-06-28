#!/bin/bash

for FOLD in $(seq 0 4)
do
    DIR=0625_5fold_img640_${FOLD}

    mkdir ${DIR}

    cp siim_${FOLD}.yml ${DIR}
    cp train.sh ${DIR}

    CUDA_VISIBLE_DEVICES=2 python train.py \
        -f siim_${FOLD}.yml \
        -e 50 \
        -b 64 \
        -p ${DIR}/ \
        -c best.pth
done
