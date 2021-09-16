#!/usr/bin/env bash

CONFIG="../configs/vfnet/vfnet_x101_32x4d_fpn_mstrain_2x_coco.py"
GPUS=2
PORT=${PORT:-29500}

#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
