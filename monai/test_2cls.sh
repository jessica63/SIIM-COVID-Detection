CUDA_VISIBLE_DEVICES=3 python3 ./test_2cls.py \
    -f siim_2cls.yml \
    -w best.pth \
    -p ./0622_fold0_2cls/ \
    -b 32 \
    -l tmp_2cls.csv \
