import os
# import sys
import argparse
import multiprocessing
import glob

import csv
import tqdm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# from ensemble_boxes import weighted_boxes_fusion as wbf

from load_dicom import save_dcm_to_img
# from load_json import read_annotation
from utils import draw_bbox


lesion_id_to_name = {
    0: "Negative",
    1: "Typical",
    2: "Indeterminate",
    3: "Atypical"
}

# color list for display bounding boxes
label2color = [
    [59, 238, 119],
    [222, 21, 229],
    [94, 49, 164],
    [206, 221, 133]
]


def df2bbox(study_df, image_df, img_shape, mode):
    one_hot = study_df[
            [
                'Negative for Pneumonia', 'Typical Appearance',
                'Indeterminate Appearance', 'Atypical Appearance'
            ]
    ].to_numpy()
    # cls = np.where(one_hot == 1)[1]
    cls = 0
    bbox = image_df['label'].values[0].split(' ')
    bbox_ls = []

    if bbox[0] == 'none':
        return bbox_ls

    for i in range(len(bbox) // 6):
        tmp_bbox = np.array(bbox[(2 + i*6):(6 + i*6)]).astype(np.float32)
        tmp_bbox /= [img_shape[1], img_shape[0], img_shape[1], img_shape[0]]
        x_min, y_min, x_max, y_max = tmp_bbox

        if mode == 'yolo':
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            w = x_max - x_min
            h = y_max - y_min

            bbox_ls.append([cls, x_center, y_center, w, h])
        else:
            bbox_ls.append([cls, x_min, y_min, x_max, y_max])

    return np.array(bbox_ls)


def processing_case(img_info_dict):
    img_id = img_info_dict["img_path"]
    save_base_dir = img_info_dict["save_base_dir"]
    save_img_size = img_info_dict["save_img_size"]

    img_dir = os.path.join(save_base_dir, img_info_dict["img_dirname"])
    csv_file = os.path.join(save_base_dir, save_img_size)

    # Save pixel data to png
    img_shape, pixel_data = save_dcm_to_img(
        dcm_path=img_id,
        save_dir=img_dir,
        force_replace=False,
        return_pixel_data=True,
        **img_info_dict["kwargs"]
    )

    with open(csv_file, 'a') as f:
        writer = csv.writer(f)
        row = [img_id.split('/')[-1][:-4], img_shape[1], img_shape[0]]
        writer.writerow(row)


def main(args):
    # npz_dirname = "img_npz"
    img_dirname = "test_clahe"
    # txt_dirname = "bbox_txt"
    clahe_args = [
        {"clipLimit": 2, "tileGridSize": (5, 5)},
        {"clipLimit": 4., "tileGridSize": (20, 20)}
    ]
    test_image_path = glob.glob(f"{args.test_dicom_dir}**/**/**.dcm")
    csv_path = os.path.join(args.save_base_dir, args.save_img_size_csv)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        column = ['img_id', 'shape_w', 'shape_h']
        writer.writerow(column)

    print("Preparing argment list for running function...")
    info_dict_list = [
        {
            "img_path": img_id,
            "save_base_dir": args.save_base_dir,
            "save_img_size": csv_path,
            "img_dirname": img_dirname,
            "kwargs": {"clahe_args": clahe_args}
        }
        for img_id in test_image_path
    ]
    os.makedirs(os.path.join(args.save_base_dir, img_dirname), exist_ok=True)

    print("Now running preprocessing part")
    with multiprocessing.Pool(args.workers) as pool:
        gen = pool.imap(processing_case, info_dict_list)
        for _ in tqdm.tqdm(gen, total=len(info_dict_list)):
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parsing raw data to processed data, "
                    "npz for images and txt for bounding boxes labels"
    )

    parser.add_argument(
        "--test-dicom-dir",
        type=str,
        default="/data2/chest_xray/siim-covid19-detection/raw/test/",
        help="Path of train dicom data directory"
    )

    parser.add_argument(
        "--save-base-dir",
        type=str,
        default="/data2/chest_xray/siim-covid19-detection/preprocessed/",
        help="Path to store preprocessed *.npz and *.txt files"
    )

    parser.add_argument(
        "--save-img-size-csv",
        type=str,
        default='img_size.csv'
        )

    parser.add_argument(
        "--workers",
        type=int,
        default=8
    )
    args = parser.parse_args()

    main(args)
