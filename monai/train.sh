#!/bin/bash

DIR=0625_EMA

mkdir ${DIR}

cp ./config/siim_0.yml ${DIR}
cp train.sh ${DIR}

CUDA_VISIBLE_DEVICES=2 python train.py \
    -f ./config/siim_0.yml \
    -e 20 \
    -b 16 \
    -p ${DIR}/ \
    -c best.pth
