#!/bin/bash

for FOLD in $(seq 0 4)
do
    DIR='0621_5fold_${FOLD}'

    mkdir ${DIR}

    cp siim.yml ${DIR}
    cp train.sh ${DIR}

    CUDA_VISIBLE_DEVICES=2 python train.py \
        -f siim_${FOLD}.yml \
        -e 50 \
        -b 64 \
        -p ${DIR}/ \
        -c best.pth
done
