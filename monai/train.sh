#!/bin/bash

CUDA_VISIBLE_DEVICE=0,1 python train.py \
    -f siim.yml \
    -e 100 \
    -b 10 \
    -p ./test \
    -c model/best.pth
