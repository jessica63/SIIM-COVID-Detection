import os
import torch
import pandas as pd

from monai.metrics import compute_roc_auc

class Evaluator():
    def __init__(self, i_step, model, data_loader, criterion, logger, mode, device):
        max_iter = len(data_loader)

        model.eval()  # Set model to evaluation mode
        losses = {k: 0. for k in criterion.loss_dict.keys()}

        metrics = {
            m: {k: 1e-12}
            for m in ['acc', 'precision', 'recall', 'f1', 'AUC']
            for k in criterion.loss_dict.keys()
        }
        tp = {k: 1e-12 for k in criterion.loss_dict.keys()}
        tn = {k: 1e-12 for k in criterion.loss_dict.keys()}
        fp = {k: 1e-12 for k in criterion.loss_dict.keys()}
        fn = {k: 1e-12 for k in criterion.loss_dict.keys()}

        # Iterate over data.
        with torch.no_grad():
            y_pred = torch.tensor([], dtype=torch.float32).to(device)
            y = torch.tensor([], dtype=torch.float32).to(device)
            for data in data_loader:
                images = data["image"].to(device)
                target = data["label"].float().to(device)
                y_pred = torch.cat([y_pred, model(images)], dim=0)
                y = torch.cat([y, target], dim=0)

            loss = criterion(y_pred, y)

            for j in range(len(criterion.tags)):
                c = criterion.tags[j]
                auc_metric = compute_roc_auc(y_pred[:, j], y[:, j])
                metrics['AUC'][c] = auc_metric
                metrics['AUC']['Total'] += auc_metric
            print(metrics['AUC'])

            y_pred = torch.argmax(y_pred, dim=1).float()
            y = y.data.float()

            # let j for index of class
            for j in range(len(criterion.tags)):
                c = criterion.tags[j]
                p = y_pred[:, j]
                t = y[:, j]

                _tp = (p * t).sum()
                _tn = ((1 - p) * (1 - t)).sum()
                _fp = (p * (1 - t)).sum()
                _fn = ((1 - p) * t).sum()

                tp[c] += _tp
                tn[c] += _tn
                fp[c] += _fp
                fn[c] += _fn
                losses[c] += criterion.loss_dict[c]

                tp['Total'] += _tp
                tn['Total'] += _tn
                fp['Total'] += _fp
                fn['Total'] += _fn

            losses['Total'] += criterion.loss_dict['Total']

            for c in list(criterion.loss_dict.keys()):
                metrics['acc'][c] = (tp[c] + tn[c]) / (tp[c] + tn[c] + fp[c] + fn[c])
                metrics['precision'][c] = (tp[c] / (tp[c] + fp[c]))
                metrics['recall'][c] = (tp[c] / (tp[c] + fn[c]))
                metrics['f1'][c] = 2 * metrics['precision'][c] * metrics['recall'][c] / (
                        metrics['precision'][c] + metrics['recall'][c] + 1)


            acc = metrics['acc']['Total']
            precision = metrics['precision']['Total']
            recall = metrics['recall']['Total']
            f1 = metrics['f1']['Total']
            total_AUC = metrics['AUC']['Total'] / len(criterion.tags)
            total_loss = losses['Total'] / max_iter

        for c in list(criterion.loss_dict.keys()):
            losses[c] /= max_iter

        print(
            f'''End Eval [Loss: total {total_loss}] [AUC: total {total_AUC}]\n[Metrics: acc {acc} precision {precision} recall {recall} f1 {f1}]'''
        )

        logger.add_scalars(f'{mode}/loss', losses, i_step)
        for m in ['acc', 'precision', 'recall', 'f1', 'AUC']:
            logger.add_scalars(f'{mode}/{m}', metrics[m], i_step)

        self.acc = acc
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.loss = total_loss
        self.auc = total_AUC
