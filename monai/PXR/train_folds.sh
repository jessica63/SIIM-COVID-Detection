#!/bin/bash

MODELS=('densenet121' 'densenet169' 'densenet201' 'densenet264')
TRAIN_DATALISTS=('bin01fold_1.json' 'bin01fold_2.json' 'bin01fold_3.json' 'bin01fold_4.json' 'bin01fold_5.json')

mkdir -p models
for d in "${TRAIN_DATALISTS[@]}"
do
    for m in "${MODELS[@]}"
    do
        python train.py \
            -d /data/datalist/bin01/$d \
            -r /data/preprocessing \
	    -e 300 \
	    -b 64 \
	    -m $m \
	    -c "models/${m}_$(basename $d .json).pth"
    done
done
