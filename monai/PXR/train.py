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
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from loss import BCELoss
from utils import Evaluator

import monai
from monai.metrics import compute_roc_auc
from monai.transforms import AddChanneld, Compose, LoadImaged
from monai.transforms import Resized, ScaleIntensityd, ToTensord
from monai.transforms import RandShiftIntensityd, RandRotated, RandFlipd, RandZoomd
from monai.transforms.compose import MapTransform
from monai.networks.nets.efficientnet import EfficientNetBN

from aug import CLAHEd

class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)

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
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=14
        )
    else:
        model = supported_models[model_name](
            spatial_dims=2,
            in_channels=in_channels,
            num_classes=14
        )
    #endif
    return model
#end


def main(args):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    params = Params(args.project_file)

    data_list   = json.load(open(params.data_list))
    num_classes = len(data_list["label_format"])

    train_files = data_list["training"]
    for f in train_files:
        f["image"] = os.path.join(params.image_dir, f["image"])
        f["label"] = torch.tensor(f["label"])
    #end

    val_files   = data_list["validation"]
    for f in val_files:
        f["image"] = os.path.join(params.image_dir, f["image"])
        f["label"] = torch.tensor(f["label"])
    #end

    print('train: ', len(train_files))
    print('valid: ', len(val_files))

    # Define transforms for image
    train_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            # CLAHEd(keys=["image"], clip_limit=4.0, tile_grid_size=(8, 8)),
            AddChanneld(keys=["image"]),
            Resized(keys=["image"], spatial_size=(400, 400)),
           #  RandRotated(keys=["image"], range_x=np.pi / 12, prob=0.5, keep_size=True),
            # RandFlipd(keys=["image"], spatial_axis=0, prob=0.5),
            # RandZoomd(keys=["image"], min_zoom=0.9, max_zoom=1.1, prob=0.5),
            #AddChanneld(keys=["image"]),
            ToTensord(keys=["image"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            # CLAHEd(keys=["image"], clip_limit=4.0, tile_grid_size=(8, 8)),
            AddChanneld(keys=["image"]),
            Resized(keys=["image"], spatial_size=(400,400)),
            #AddChanneld(keys=["image"]),
            ToTensord(keys=["image"]),
        ]
    )
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=torch.cuda.is_available()
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
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # device = torch.device(tuple(args.gpu))
    device = torch.device("cuda:3")# if torch.cuda.is_available() else "cpu")
    # model = get_model_from_name(args.model, in_channels=1).to(device)
    model = EfficientNetBN("efficientnet-b4", spatial_dims=2, in_channels=1, num_classes=4)
    model = torch.nn.DataParallel(model, device_ids=[3, 1,2])
    loss_function = BCELoss(params.obj_list, params.stats)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epoch,
        eta_min=1e-5
    )

    # start a typical PyTorch training
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    writer = SummaryWriter(os.path.join(args.prefix, 'tensorboard'))
    for epoch in range(args.epoch):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{args.epoch}")
        model.train()
        epoch_loss = 0
        step = 0

        for batch_data in train_loader:
            step += 1
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("running/loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        scheduler.step()

        if (epoch + 1) % val_interval == 0:

            train_evaluator = Evaluator(
                epoch_len * epoch + step,
                model,
                train_loader,
                loss_function,
                writer,
                'train',
                device
            )
            valid_evaluator = Evaluator(
                epoch_len * epoch + step,
                model,
                val_loader,
                loss_function,
                writer,
                'valid',
                device
            )

            if valid_evaluator.auc > best_metric:
                best_metric = valid_evaluator.auc
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(args.prefix,f"{best_metric_epoch}_model.ckpt"))
                print("saved new best metric model")
            print(
                "current epoch: {} current AUC: {:.4f} best AUC: {:.4f} at epoch {}".format(
                    epoch + 1, valid_evaluator.auc, best_metric, best_metric_epoch
                )
            )
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--project_file", type=str, help="project configuration")
    # parser.add_argument("-m", "--model", type=str, help="model name")
    parser.add_argument("-e", "--epoch", type=int, help="number of training epochs")
    parser.add_argument("-b", "--batch_size", type=int, help="training batch size")
    parser.add_argument("-p", "--prefix", type=str, help="prefix path")
    parser.add_argument("-g", "--gpu", type=str, help="which GPU")
    args = parser.parse_args()

    main(args)
