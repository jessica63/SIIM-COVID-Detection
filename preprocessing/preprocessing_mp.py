import os
# import sys
import argparse
import multiprocessing
import glob

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


def bboxes_fusion(img_shape, img_bbox_df):
    bbox_coord_col = ["x_min", "y_min", "x_max", "y_max"]
    # bboxes fusion (WBF)
    class_ids = img_bbox_df["class_id"].values
    class_id_cnt = img_bbox_df["class_id"].value_counts().to_dict()
    orig_bboxes = img_bbox_df[bbox_coord_col].values
    # TODO: check which axis is x-axis or y-axis
    # Normalize the bboxes coordinate
    orig_bboxes /= [img_shape[1], img_shape[0], img_shape[1], img_shape[0]]
    orig_bboxes = np.clip(orig_bboxes, 0, 1)
    fused_bboxes = []
    fused_id = []

    for c_id in class_id_cnt.keys():
        if c_id == 14:
            continue
        if class_id_cnt[c_id] == 1:  # only one bboxes
            selected_bbox = np.array(orig_bboxes[class_ids == c_id])
            # nothing to do with only single bbox
            assert selected_bbox.shape[0] == 1, "Allowed only one bbox"
            fused_bboxes.append(selected_bbox)
            fused_id.append([c_id])

        else:  # more than two bboxes
            selected_bboxes = np.array(orig_bboxes[class_ids == c_id])
            assert selected_bboxes.shape[0] == class_id_cnt[c_id]
            # Use weighted boxes fusion to fuse bboxes
            wbf_boxes, _, labels = wbf(
                boxes_list=[selected_bboxes],
                scores_list=np.ones((1, class_id_cnt[c_id])),
                labels_list=np.full((1, class_id_cnt[c_id]), fill_value=c_id),
                iou_thr=1,
                skip_box_thr=1e-3
            )
            fused_bboxes.append(wbf_boxes)
            fused_id.append(labels)

    if len(fused_bboxes) == 0:
        return fused_bboxes

    # Concate the bboxes together and save to txt file
    fused_bboxes = np.concatenate(fused_bboxes)
    fused_id = np.concatenate(fused_id).astype(np.int)
    fused_labels = np.hstack([np.expand_dims(fused_id, -1), fused_bboxes])
    return fused_labels


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
    img_id = img_info_dict["img_id"]
    study_df = img_info_dict["study_df"]
    image_df = img_info_dict["image_df"]
    train_dcm_dir = img_info_dict["train_dcm_dir"]
    save_base_dir = img_info_dict["save_base_dir"]

    img_dir = os.path.join(save_base_dir, img_info_dict["img_dirname"])
    bbox_dir = os.path.join(save_base_dir, img_info_dict["txt_dirname"])

    # Save pixel data to png
    img_shape, pixel_data = save_dcm_to_img(
        dcm_path=glob.glob(f"{train_dcm_dir}**/**/{img_id}.dcm")[0],
        save_dir=img_dir,
        force_replace=False,
        return_pixel_data=True,
        **img_info_dict["kwargs"]
    )

    # Create bboxes
    labels = df2bbox(study_df, image_df, img_shape, mode='yolo')

    txt_fpath = os.path.join(bbox_dir, img_id + ".txt")
    save_fmt = "%d" + " %.7f" * 4 if len(labels) > 0 else "%d"
    np.savetxt(
        fname=txt_fpath,
        X=labels,
        fmt=save_fmt
    )

    # Display and draw bounding boxes
    if "display_dirname" in img_info_dict.keys():
        dis_dir = os.path.join(save_base_dir, img_info_dict["display_dirname"])
        pixel_data = (pixel_data * 255.).astype(np.uint8)
        rgb_img = np.repeat(pixel_data[..., np.newaxis], axis=-1, repeats=3)

        if len(labels) > 0:
            labels[:, [1, 3]] *= img_shape[1]
            labels[:, [2, 4]] *= img_shape[0]
            for bbox in labels:
                label_id = int(bbox[0])
                color = label2color[label_id]
                label_name = lesion_id_to_name[label_id]
                x, y, w, h = bbox[1:]
                box = np.array([
                        x - (w / 2),
                        y - (h / 2),
                        x + (w / 2),
                        y + (h / 2)
                    ]
                ).astype(np.int)
                rgb_img = draw_bbox(
                    rgb_img,
                    box=list(box),
                    label=label_name,
                    color=color
                )

        plt.imsave(os.path.join(dis_dir, img_id + ".jpg"), rgb_img)
        # end


def main(args):
    study_df = pd.read_csv(args.study_csv)
    image_df = pd.read_csv(args.image_csv)

    # npz_dirname = "img_npz"
    img_dirname = "clahe_images_640"
    txt_dirname = "clahe_bbox_txt"
    clahe_args = [
        {"clipLimit": 2, "tileGridSize": (5, 5)},
        {"clipLimit": 4., "tileGridSize": (20, 20)}
    ]

    print("Preparing argment list for running function...")
    info_dict_list = [
        {
            "img_id": img_id.split('_')[0],
            "study_df": study_df[study_df['id'] == study_id+'_study'],
            "image_df": img_image_df,
            "train_dcm_dir": args.train_dicom_dir,
            "save_base_dir": args.save_base_dir,
            "img_dirname": img_dirname,
            "txt_dirname": txt_dirname,
            "kwargs": {"clahe_args": clahe_args}
        }
        for img_id, img_image_df in image_df.groupby(by="id")
        for study_id in img_image_df['StudyInstanceUID']
    ]
    os.makedirs(os.path.join(args.save_base_dir, img_dirname), exist_ok=True)
    os.makedirs(os.path.join(args.save_base_dir, txt_dirname), exist_ok=True)

    # Whether to save display *.jpg
    if args.save_display:
        display_dirname = "display"
        os.makedirs(
            os.path.join(args.save_base_dir, display_dirname),
            exist_ok=True
        )
        for info_dict in info_dict_list:
            info_dict["display_dirname"] = display_dirname

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
        "--train-dicom-dir",
        type=str,
        default="/data2/chest_xray/siim-covid19-detection/raw/train/",
        help="Path of train dicom data directory"
    )

    parser.add_argument(
        "--study-csv",
        type=str,
        default="/data2/chest_xray/siim-covid19-detection/raw/train_study_level.csv",
        help="Path of train study level label directory"
    )

    parser.add_argument(
        "--image-csv",
        type=str,
        default="/data2/chest_xray/siim-covid19-detection/raw/train_image_level.csv",
        help="Path of train image level label directory"
    )

    parser.add_argument(
        "--save-base-dir",
        type=str,
        default="/data2/chest_xray/siim-covid19-detection/preprocessed/",
        help="Path to store preprocessed *.npz and *.txt files"
    )

    parser.add_argument(
        "--save-display",
        action="store_true",
        default=False,
        help="Saving image with displaying bounding boxes "
             "to <SAVE_BASE_DIR>/display"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8
    )
    args = parser.parse_args()

    main(args)
