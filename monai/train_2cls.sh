#!/bin/bash

#for FOLD in $(seq 0 1)
#do
    FOLD=0
    DIR=0622_fold${FOLD}_2cls

    mkdir ${DIR}

    cp siim_2cls.yml ${DIR}
    cp train_2cls.sh ${DIR}

    CUDA_VISIBLE_DEVICES=2 python train_2cls.py \
        -f siim_2cls.yml \
        -e 50 \
        -b 64 \
        -p ${DIR}/ \
        -c best.pth
#done
