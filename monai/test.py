# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import json
import logging
import argparse
import yaml

import tqdm
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.metrics import compute_roc_auc, get_confusion_matrix
from monai.transforms import Activations, AsDiscrete, RandFlipd, RandRotated
from monai.transforms import AddChanneld, Compose, LoadImaged
from monai.transforms import Resized, ScaleIntensityRanged, ToTensord
from monai.transforms import RandShiftIntensityd, AsChannelFirstd
from monai.networks.nets.efficientnet import EfficientNetBN

from sklearn.metrics import average_precision_score, confusion_matrix

from utils import plot_confusion_matrix


class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


def main(args):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    params = Params(os.path.join(args.prefix, args.project_file))

    data_list = json.load(open(params.data_list))
    num_cls = len(data_list["label_format"])

    val_files = data_list["validation"]
    for f in val_files:
        f["image"] = os.path.join(params.image_dir, f["image"])
        f["label"] = torch.tensor(f["label"])
    # end

    print('valid: ', len(val_files))

    val_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            # AddChanneld(keys=["image"]),
            AsChannelFirstd(keys=["image"]),
            Resized(keys=["image"], spatial_size=(400, 400)),
            ToTensord(keys=["image"]),
        ]
    )

    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNetBN(
            "efficientnet-b4",
            spatial_dims=2,
            in_channels=3,
            num_classes=num_cls
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(args.prefix, args.weight)))
    model.eval

    act = Activations(softmax=True)
    to_onehot = AsDiscrete(to_onehot=True, n_classes=4)

    # start a typical PyTorch training
    with torch.no_grad():
        y_pred = torch.tensor([], dtype=torch.float32, device=device)
        y = torch.tensor([], dtype=torch.long, device=device)
        for val_data in val_loader:
            val_images = val_data["image"].to(device)
            val_labels = torch.argmax(
                    val_data["label"].to(device),
                    dim=1
            )
            y_pred = torch.cat([y_pred, model(val_images)], dim=0)
            y = torch.cat([y, val_labels], dim=0)

        y_onehot = to_onehot(y)
        y_pred_onehot = act(y_pred)

        if args.save_result:
            df = pd.DataFrame(
                {
                    "UID": f["image"],
                    "Negative": y_pred[:, 0],
                    "Typical": y_pred[:, 1],
                    "Indeterminate": y_pred[:, 2],
                    "Atypical": y_pred[:, 3],
                    "class": y_pred_onehot
                }
                for f in val_files
            )
            df.to_csv(os.path.join(args.prefix, "test_result.csv"))

        auc_metric = compute_roc_auc(y_pred_onehot, y_onehot)
        cm = confusion_matrix(y_onehot.argmax(dim=1).cpu(), y_pred_onehot.argmax(dim=1).cpu())
        aps = []
        for i in range(num_cls):
            ap = average_precision_score(y_onehot.cpu()[:, i], y_pred_onehot.cpu()[:, i])
            aps.append(ap)

        log_file = args.log
        if not os.path.exists(log_file):
            write_col = True
        else:
            write_col = False

        with open(log_file, "a") as log:
            writer = csv.writer(log)
            if write_col:
                writer.writerow([
                    "prefix",
                    "AUC",
                    "AP",
                    "AP_Negative",
                    "AP_Typical",
                    "AP_Indeterminate",
                    "AP_Atypical"
                ])
            writer.writerow(
                [
                    args.prefix,
                    auc_metric,
                    np.array(aps).mean(),
                    aps[0],
                    aps[1],
                    aps[2],
                    aps[3]
                ]
            )
        print("auc:", auc_metric)
        print("cm: ", cm, cm.shape)
        print("AP: ", np.array(aps).mean(), aps)
        plt.figure()
        plot_confusion_matrix(cm, data_list["label_format"])
        plt.savefig(f"{args.prefix}/confusion_metric.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--project_file", type=str, help="project configuration")
    parser.add_argument("-w", "--weight", type=str, help="path to weight")
    parser.add_argument("-b", "--batch_size", type=int, help="training batch size")
    parser.add_argument("-p", "--prefix", type=str, help="prefix path")
    parser.add_argument("-l", "--log", type=str, help="test log file")
    parser.add_argument("--save_result", action='store_true', help="save result")
    args = parser.parse_args()

    main(args)
