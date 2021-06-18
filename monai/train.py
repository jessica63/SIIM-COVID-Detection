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

import monai
from monai.metrics import compute_roc_auc
from monai.transforms import Activations, AsDiscrete
from monai.transforms import AddChanneld, Compose, LoadImaged
from monai.transforms import Resized, ScaleIntensityRanged, ToTensord
from monai.transforms import RandShiftIntensityd
from monai.networks.nets.efficientnet import EfficientNetBN


class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


def main(args):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    params = Params(args.project_file)

    data_list = json.load(open(params.data_list))
    num_cls = len(data_list["label_format"])

    train_files = data_list["training"]
    for f in train_files:
        f["image"] = os.path.join(params.image_dir, f["image"])
        f["label"] = torch.tensor(f["label"])
    # end

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
            AddChanneld(keys=["image"]),
            Resized(keys=["image"], spatial_size=(400, 400)),
            ToTensord(keys=["image"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Resized(keys=["image"], spatial_size=(400, 400)),
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNetBN(
            "efficientnet-b4",
            spatial_dims=2,
            in_channels=1,
            num_classes=num_cls
    ).to(device)
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5)
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
            labels = torch.argmax(batch_data["label"].to(device), dim=1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            train_acc = np.where(
                    outputs.argmax(dim=1).cpu().detach().numpy() ==
                    labels.cpu().detach().numpy(), 1.0, 0.0
            ).mean()
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}, train_acc: {train_acc:.4f}")
            writer.add_scalar("running/loss", loss.item(), epoch_len * epoch + step)
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
                    val_labels = torch.argmax(
                            val_data["label"].to(device),
                            dim=1
                    )
                    y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                    y = torch.cat([y, val_labels], dim=0)

                valid_loss = loss_function(y_pred, y)
                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)

                y_onehot = to_onehot(y)
                y_pred_onehot = act(y_pred)
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
                writer.add_scalar("valid/val_loss", valid_loss, epoch + 1)
                writer.add_scalar("valid/val_accuracy", acc_metric, epoch + 1)
                writer.add_scalar("valid/val_AUC", auc_metric, epoch + 1)
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--project_file", type=str, help="project configuration")
    parser.add_argument("-e", "--epoch", type=int, help="number of training epochs")
    parser.add_argument("-b", "--batch_size", type=int, help="training batch size")
    parser.add_argument("-p", "--prefix", type=str, help="prefix path")
    parser.add_argument("-c", "--ckpt", type=str, help="checkpoint file name")
    parser.add_argument("-g", "--gpu", type=str, help="which GPU")
    args = parser.parse_args()

    main(args)
