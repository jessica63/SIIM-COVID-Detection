CUDA_VISIBLE_DEVICES=2 python3 ./test.py \
    -f siim.yml \
    -w best.pth \
    -p ./0621_focal/ \
    -b 32 \
    -l tmp.csv \
