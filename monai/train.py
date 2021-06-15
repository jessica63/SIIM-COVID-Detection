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

import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.metrics import compute_roc_auc
from monai.transforms import Activations, AsDiscrete
from monai.transforms import AddChanneld, Compose, LoadImaged
from monai.transforms import Resized, ScaleIntensityRanged, ToTensord
from monai.transforms import RandShiftIntensityd, Rand3DElasticd
from monai.transforms.compose import MapTransform
from monai.networks.nets.efficientnet import EfficientNetBN

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
        "se_resnext101_32x4d": monai.networks.nets.senet.se_resnext101_32x4d,
        "efficientnet_b5": monai.networks.nets.efficientnet.efficientnet-b5,
        "efficientnet_b6": monai.networks.nets.efficientnet.efficientnet-b6,
        "efficientnet_b7": monai.networks.nets.efficientnet.efficientnet-b7,
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

class Concatenate(MapTransform):
    def __init__(self, keys, out_key, axis):
        self.keys    = keys
        self.out_key = out_key
        self.axis    = axis
    #end

    def __call__(self, data):
        data[self.out_key] = np.concatenate([data[k] for k in self.keys], axis=self.axis)
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

def main(args, img_size=(600, 600)):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    data_list   = json.load(open(args.data_list))
    num_classes = len(data_list["label_format"])

    train_files = data_list["training"]
    for f in train_files:
        f["image"] = os.path.join(args.data_root, f["image"])
        f["label"] = torch.tensor(f["label"])
    #end

    val_files   = data_list["validation"]
    for f in val_files:
        f["image"] = os.path.join(args.data_root, f["image"])
        f["label"] = torch.tensor(f["label"])
    #end

    # Define transforms for image
    train_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            #ScaleIntensityRanged(
            #    keys=["image"],
            #    a_min=0,
            #    a_max=256,
            #    b_min=0.0,
            #    b_max=1.0,
            #    clip=True
            #),
            Resized(keys=["image"], spatial_size=img_size),
            # AddChanneld(keys=["image"]),
            ToTensord(keys=["image"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
           #  ScaleIntensityRanged(
                # keys=["image"],
                # a_min=0,
                # a_max=256,
                # b_min=0.0,
                # b_max=1.0,
                # clip=True
            # ),
            Resized(keys=["image"], spatial_size=img_size),
            ToTensord(keys=["image"]),
        ]
    )

    # create a training data loader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=torch.cuda.is_available()
    )

    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(
        val_ds,
        batch_size=24,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNetBN("efficientnet-b4", spatial_dims=2, in_channels=1, num_classes=4).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epoch,
        eta_min=1e-5
    )
    act = Activations(softmax=True)
    to_onehot = AsDiscrete(to_onehot=True, n_classes=4)

    # start a typical PyTorch training
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    writer = SummaryWriter()
    for epoch in range(args.epoch):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{args.epoch}")
        model.train()
        epoch_loss = 0
        step = 0

        for batch_data in train_loader:
            step += 1
            inputs = batch_data["image"].to(device)
            labels = torch.argmax(batch_data["label"].to(device), dim=1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            train_acc = np.where(outputs.argmax(dim=1).cpu().detach().numpy() == labels.cpu().detach().numpy(), 1.0, 0.0).mean()
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}, train_acc: {train_acc:.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        scheduler.step()

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                for val_data in val_loader:
                    val_images = val_data["image"].to(device)
                    val_labels = torch.argmax(val_data["label"].to(device), dim=1)
                    y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                    y = torch.cat([y, val_labels], dim=0)

                y_onehot = to_onehot(y)
                y_pred_onehot = act(y_pred)
                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)
                auc_metric = compute_roc_auc(y_pred_onehot, y_onehot)
                if auc_metric > best_metric:
                    best_metric = auc_metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), f"{args.ckpt}")
                    print("saved new best metric model")
                print(
                    "current epoch: {} current accuracy: {:.4f} current AUC: {:.4f} best AUC: {:.4f} at epoch {}".format(
                        epoch + 1, acc_metric, auc_metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_accuracy", acc_metric, epoch + 1)
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_list", type=str, help="data list for training and validation")
    parser.add_argument("-r", "--data_root", type=str, help="root directory of data")
    parser.add_argument("-m", "--model", type=str, help="model name")
    parser.add_argument("-e", "--epoch", type=int, help="number of training epochs")
    parser.add_argument("-b", "--batch_size", type=int, help="training batch size")
    parser.add_argument("-c", "--ckpt", type=str, help="checkpoint file name")
    parser.add_argument("-g", "--gpu", type=str, help="which GPU to use")
    args = parser.parse_args()

    main(args, img_size=(400, 400))
