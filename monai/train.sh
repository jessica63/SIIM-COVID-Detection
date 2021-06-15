#!/bin/bash

python train.py \
    -d /data/mask_data_list_training.json \
    -r /data/preprocessing \
    -e 200 \
    -b 48 \
    -m densenet201 \
    -c models/densenet201.pth
