import torch
import torch.nn as nn
import torch.nn.functional as F

class BCELoss(nn.Module):
    """docstring for BCELoss"""
    def __init__(self, tags, stats):
        super(BCELoss, self).__init__()
        self.tags = tags
        self.stats = torch.tensor(stats).cuda()
        self.loss_dict = {k: 0. for k in self.tags + ['Total']}

    def forward(self, pred, target):

        stats = self.stats.expand_as(target)
        weight = torch.abs(target - stats)

        total_loss = 0
        for i in range(len(self.tags)):
            self.loss_dict[self.tags[i]] = F.binary_cross_entropy_with_logits(
                pred[:, i],
                target[:, i],
                weight[:, i]
            )
            total_loss += self.loss_dict[self.tags[i]]
        self.loss_dict['Total'] = total_loss/len(self.tags)
        return self.loss_dict['Total']
