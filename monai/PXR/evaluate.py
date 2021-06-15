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

import numpy as np
import torch
from torch.utils.data import DataLoader

import monai
from monai.data import CSVSaver
from monai.metrics import compute_roc_auc
from monai.transforms import AddChanneld, Compose, LoadNiftid
from monai.transforms import Resized, ScaleIntensityRanged, ToTensord
from monai.transforms.compose import MapTransform

def get_model_from_name(model_name, in_channels=1):
    supported_models = {
        "densenet121": monai.networks.nets.densenet.densenet121,
        "densenet169": monai.networks.nets.densenet.densenet169,
        "densenet201": monai.networks.nets.densenet.densenet201,
        "densenet264": monai.networks.nets.densenet.densenet264,
        "senet154": monai.networks.nets.senet.senet154,
        "se_resnet50": monai.networks.nets.senet.se_resnet50,
        "se_resnet101": monai.networks.nets.senet.se_resnet101,
        "se_resnet152": monai.networks.nets.senet.se_resnet152,
        "se_resnext50_32x4d": monai.networks.nets.senet.se_resnext50_32x4d,
        "se_resnext101_32x4d": monai.networks.nets.senet.se_resnext101_32x4d
    }

    if model_name not in supported_models.keys():
        raise RuntimeError("Invalid model name")
    #endif

    if "densenet" in model_name:
        model = supported_models[model_name](
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=2
        )
    else:
        model = supported_models[model_name](
            spatial_dims=3,
            in_channels=in_channels,
            num_classes=2
        )
    #endif
    return model
#end

class ApplyMaskd(MapTransform):
    def __init__(self, keys, mask_key):
        self.keys     = keys
        self.mask_key = mask_key
    #end

    def __call__(self, data):
        mask = np.clip(data[self.mask_key].astype(np.float), 0.0, 1.0)
        for k in self.keys:
            data_array = mask * data[k]
            data[k] = data_array
        #end
        return data
    #end
#end

class LoadImage(MapTransform):
    def __init__(self, img_path):
        self.img_path    = img_path
    #end

    def __call__(self, data):
        data[self.out_key] = np.concatenate([data[k] for k in self.keys], axis=self.axis)
        return data
    #end
#end


def main(args, crop_size=(96, 96, 96)):
    # monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    data_list   = json.load(open(args.data_list))
    num_classes = len(data_list["label_format"])

    val_files = data_list[args.data_key]
    for f in val_files:
        f["image"] = os.path.join(args.data_root, f["image"])
    #end

    # Define transforms for image
    val_transforms = Compose(
        [
            LoadImaged(keys=["image"], npz_key = ["img"]),
            AddChanneld(keys=["image"]),
            ScaleIntensityd(keys=["image"]),
            Resized(keys=["image"], spatial_size=crop_size),
            ToTensord(keys=["image", "label"]),
        ]
    )

    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=4, pin_memory=torch.cuda.is_available())

    # Create DenseNet121
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model_from_name(args.model, in_channels=1).to(device)

    model.load_state_dict(torch.load(args.ckpt))
    model.eval()
    with torch.no_grad():
        num_correct = 0.0
        metric_count = 0
        val_y     = torch.tensor([], dtype=torch.float32, device=device)
        val_preds = torch.tensor([], dtype=torch.float32, device=device)
        for val_data in val_loader:
            val_images  = val_data["image"].to(device)
            val_labels  = torch.argmax(val_data["label"], dim=1).to(device)
            val_outputs = model(val_images)

            val_y     = torch.cat([val_y, val_labels], dim=0)
            val_preds = torch.cat([val_preds, val_outputs], dim=0)

            val_outputs = val_outputs.argmax(dim=1)
            value = torch.eq(val_outputs, val_labels)
            metric_count += len(value)
            num_correct += value.sum().item()
        acc = num_correct / metric_count
        auc = compute_roc_auc(val_preds, val_y, to_onehot_y=True, softmax=True)
        print(f"evaluation accuracy: {acc}, auc: {auc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_list", type=str, help="data list for validation")
    parser.add_argument("-r", "--data_root", type=str, help="root directory of data")
    parser.add_argument("-k", "--data_key", type=str, help="dataset key in data list file")
    parser.add_argument("-c", "--ckpt", type=str, help="checkpoint file")
    parser.add_argument("-m", "--model", type=str, help="model name")
    args = parser.parse_args()

    main(args, crop_size=(96, 96, 96))
