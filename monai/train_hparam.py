import os
import argparse

import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, help="gpu device id")
    args = parser.parse_args()

    args.data_list = "/data/mask_data_list_training.json"
    args.data_root = "/data/preprocessing"
    args.model     = "densenet201"

    # Exp 1
    args.epoch      = 200
    args.batch_size = 32
    args.ckpt       = "models/densenet201_64_64_32.pth"
    train.main(args, crop_size=(64, 64, 32))

    # Exp 2
    args.epoch      = 200
    args.batch_size = 32
    args.ckpt       = "models/densenet201_64_64_64.pth"
    train.main(args, crop_size=(64, 64, 64))

    # Exp 3
    args.epoch      = 200
    args.batch_size = 32
    args.ckpt       = "models/densenet201_96_64_32.pth"
    train.main(args, crop_size=(96, 64, 32))

    # Exp 4
    args.epoch      = 200
    args.batch_size = 32
    args.ckpt       = "models/densenet201_64_96_32.pth"
    train.main(args, crop_size=(64, 96, 32))

    # Exp 5
    args.epoch      = 200
    args.batch_size = 32
    args.ckpt       = "models/densenet201_96_96_32.pth"
    train.main(args, crop_size=(96, 96, 32))

    # Exp 5
    args.epoch      = 200
    args.batch_size = 32
    args.ckpt       = "models/densenet201_96_96_64.pth"
    train.main(args, crop_size=(96, 96, 64))

    # Exp 6
    args.epoch      = 200
    args.batch_size = 32
    args.ckpt       = "models/densenet201_96_64_96.pth"
    train.main(args, crop_size=(96, 64, 96))

    # Exp 7
    args.epoch      = 200
    args.batch_size = 32
    args.ckpt       = "models/densenet201_64_96_96.pth"
    train.main(args, crop_size=(64, 96, 96))

    # Exp 8
    args.epoch      = 200
    args.batch_size = 32
    args.ckpt       = "models/densenet201_96_96_96.pth"
    train.main(args, crop_size=(96, 96, 96))

    # Exp 9
    args.epoch      = 200
    args.batch_size = 32
    args.ckpt       = "models/densenet201_128_128_32.pth"
    train.main(args, crop_size=(128, 128, 32))

    # Exp 10
    args.epoch      = 200
    args.batch_size = 32
    args.ckpt       = "models/densenet201_128_128_64.pth"
    train.main(args, crop_size=(128, 128, 64))

    # Exp 11
    args.epoch      = 200
    args.batch_size = 24
    args.ckpt       = "models/densenet201_128_128_96.pth"
    train.main(args, crop_size=(128, 128, 96))

    # Exp 12
    args.epoch      = 200
    args.batch_size = 24
    args.ckpt       = "models/densenet201_128_128_128.pth"
    train.main(args, crop_size=(128, 128, 128))
