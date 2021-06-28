CUDA_VISIBLE_DEVICES=3 python3 ./test.py \
    -f siim_0.yml \
    -w best.pth \
    -p ./0625_EMA/ \
    -b 16 \
    -l tmp.csv \
