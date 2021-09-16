#!/usr/bin/env bash

CONFIG="../configs/vfnet/vfnet_x101_32x4d_fpn_mstrain_2x_coco.py"
CHECKPOINT="/data2/smarted/PXR/code/jessica/vfnet_0521/latest.pth"
GPUS=1
PORT=${PORT:-29501}

#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} \
    --out /data2/smarted/PXR/code/jessica/vfnet_0521/valid.pkl
