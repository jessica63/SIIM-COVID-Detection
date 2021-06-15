#!/bin/bash

CUDA_VISIBLE_DEVICE=1,2,3 python train.py \
    -f siim.yml \
    -e 100 \
    -b 3 \
    -p ./test \
    -g 2,3
