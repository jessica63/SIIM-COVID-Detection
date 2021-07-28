import os
import seaborn as sns
sns.set(rc={"font.size":9,"axes.titlesize":15,"axes.labelsize":9,
            "axes.titlepad":11, "axes.labelpad":9, "legend.fontsize":7,
            "legend.title_fontsize":7, 'axes.grid' : False})
import argparse
import cv2
import json
import datetime
import glob
import numpy as np
from tqdm.auto import tqdm
from ensemble_boxes import *
from collections import Counter
import pydicom

labels = ["opacity"]

label2color = [[59, 238, 119]]
thickness = 3


def write_annotation(i, boxes, data_train):
    for box in boxes:
        x, y, w, h = (box[0], box[1], box[2], box[3])
        area = round(w * h, 1)
        bbox = [
            round((x - w/2), 1),
            round((y - h/2), 1),
            round(w, 1),
            round(h, 1)
        ]

        data_train['annotations'].append(dict(
            id=len(data_train['annotations']),
            image_id=i,
            category_id=0,
            area=area,
            bbox=bbox,
            iscrowd=0
        ))

    return data_train


def main(args):
    image_dir = args.image_dir
    label_dir = args.label_dir
    image_ls = os.listdir(image_dir)

    fold_info = json.load(open(args.fold_info))
    with open("/home/u/jessica63/projs/SIIM-COVID-Detection/entry/detection/valid_list_0.txt") as f:
        l = f.readlines()
    val_ls = [t[:-5].split('/')[-1] for t in l]
    print(val_ls[0])
    train = []
    valid = []
    for n in image_ls:
        if n[:-4] in val_ls:
            valid.append(n)
        else:
            train.append(n)

    now = datetime.datetime.now()

    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime('%Y-%m-%d %H:%M:%S.%f'),
        ),
        licenses=[dict(
            url=None,
            id=0,
            name=None,
        )],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type='instances',
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )

    out_file = f"{args.save_path}_{args.fold}.json"

    data_train = data.copy()
    data_train['images'] = []
    data_train['annotations'] = []

    for i in range(len(labels)):
        data_train['categories'].append({'id': i, 'name': labels[i]})

    for i, img in tqdm(enumerate(train)):
        # Add Images to annotation
        y, x = cv2.imread(os.path.join(image_dir, img)).shape[:2]
        data_train['images'].append(dict(
            license=0,
            url=None,
            file_name=os.path.join(image_dir, img),
            height=y,
            width=x,
            date_captured=None,
            id=i
        ))

        # Add bboxes to annotation
        label = img.replace(".png", ".txt")
        boxes = []
        with open(os.path.join(label_dir, label), 'r') as f:
            line = f.readlines()
            if len(line) == 0:
                continue
            boxes = [box.strip("\n").split(' ')[1:] for box in line]
            boxes = np.array(boxes).astype(np.float)
            boxes *= (x, y, x, y)
            boxes = boxes.round().tolist()

        data_train = write_annotation(i, boxes, data_train)

    with open(out_file, 'w') as f:
        json.dump(data_train, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image_dir",
        type=str,
        default="/work/Lung/SIIM/preprocessed/clahe/images/"
    )
    parser.add_argument(
        "--label_dir",
        type=str,
        default='/work/Lung/SIIM/preprocessed/clahe/labels/'
    )
    parser.add_argument(
        "--fold_info",
        type=str,
        default='/home/u/jessica63/projs/SIIM-COVID-Detection/entry/5fold_info.json',
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default='./coco_train'
    )

    args = parser.parse_args()

    main(args)
