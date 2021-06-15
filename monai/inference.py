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

def get_model_from_name(model_name):
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
            in_channels=1,
            out_channels=2
        )
    else:
        model = supported_models[model_name](
            spatial_dims=3,
            in_channels=1,
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

class DeleteKeyd(MapTransform):
    def __init__(self, keys):
        self.keys = keys
    #end

    def __call__(self, data):
        for k in self.keys:
            data.pop(k, None)
        #end
        return data
    #end
#end

def main(args):
    # monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    data_list   = json.load(open(args.data_list))
    num_classes = len(data_list["label_format"])

    val_files = data_list[args.data_key]
    for f in val_files:
        f["image"] = os.path.join(args.data_root, f["image"])
        f["mask"]  = os.path.join(args.data_root, f["mask"])
    #end

    # Define transforms for image
    val_transforms = Compose(
        [
            LoadNiftid(keys=["image", "mask"]),
            AddChanneld(keys=["image"]),
            ApplyMaskd(keys=["image"], mask_key="mask"),
            DeleteKeyd(keys=["mask"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-250,
                a_max=200,
                b_min=0.0,
                b_max=1.0,
                clip=True
            ),
            Resized(keys=["image"], spatial_size=(96, 96, 96)),
            ToTensord(keys=["image"]),
        ]
    )

    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(
        val_ds,
        batch_size=16,
        shuffle=False,
        num_workers=8,
        pin_memory=torch.cuda.is_available()
    )

    # Create DenseNet121
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model_from_name(args.model).to(device)

    model.load_state_dict(torch.load(args.ckpt))
    model.eval()
    with torch.no_grad():
        val_preds = torch.tensor([], dtype=torch.float32, device=device)
        for i, val_data in enumerate(val_loader):
            val_images  = val_data["image"].to(device)
            val_outputs = model(val_images)

            val_preds = torch.cat([val_preds, val_outputs], dim=0)
        #end

    val_preds = val_preds.cpu().detach().numpy()
    val_cases = [f["image"] for f in val_files]
    with open(args.output, "w") as f:
        f.write("Case,Pred\n")
        for c, p in zip(val_cases, val_preds):
            score = np.exp(p[1]) / np.sum(np.exp(p))
            f.write(f"{c},{score}\n")
        #end
    #end
#end


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_list", type=str, help="data list for validation")
    parser.add_argument("-r", "--data_root", type=str, help="root directory of data")
    parser.add_argument("-k", "--data_key", type=str, help="dataset key in data list file")
    parser.add_argument("-c", "--ckpt", type=str, help="checkpoint file")
    parser.add_argument("-m", "--model", type=str, help="model name")
    parser.add_argument("-o", "--output", type=str, help="output report path")
    args = parser.parse_args()

    main(args)
