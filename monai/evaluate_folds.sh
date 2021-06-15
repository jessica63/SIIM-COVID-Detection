#!/bin/bash

MODELS=('densenet121' 'densenet169' 'densenet201' 'densenet264')
TRAIN_SPLITS=('bin01fold_1' 'bin01fold_2' 'bin01fold_3' 'bin01fold_4' 'bin01fold_5')

mkdir -p results
for m in "${MODELS[@]}"
do
    for d in ${TRAIN_SPLITS[@]}
    do
        python inference.py -d /data/datalist/bin01/bin01test.json -r /data/crop_bin -k validation -c models/${m}_${d}.pth -m $m -o results/${m}_${d}.csv
        python evaluate.py  -d /data/datalist/bin01/bin01test.json -r /data/crop_bin -k validation -c models/${m}_${d}.pth -m $m  > results/${m}_${d}.txt
    done
done
