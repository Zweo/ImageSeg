import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import numpy as np


class DiceLoss(_Loss):
    def forward(self, output, target, weights=None, ignore_index=None):
        eps = 0.001

        encoded_target = output.detach() * 0  # 将variable参数从网络中隔离开，不参与参数更新。

        if ignore_index is not None:
            mask = target == ignore_index
            target = target.clone()
            target[mask] = 0
            encoded_target.scatter_(1, target.unsqueeze(1),
                                    1)  # unsqueeze增加一个维度
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0

        else:
            encoded_target.scatter_(1, target.unsqueeze(1),
                                    1)  # unsqueeze增加一个维度
            # scatter_(dim, index, src)将src中数据根据index中的索引按照dim的方向输出。

        if weights is None:
            weights = 1
        # print(output.min(),output.max())
        # print(output.shape,output[0,:,0,0])
        intersection = output * encoded_target
        numerator = 2 * intersection.sum(0).sum(1).sum(1)
        denominator = output + encoded_target

        if ignore_index is not None:
            denominator[mask] = 0

        # 计算无效的类别数量
        count1 = []
        for i in encoded_target.sum(0).sum(1).sum(1):
            if i == 0:
                count1.append(1)
            else:
                count1.append(0)
        count2 = []
        for i in denominator.sum(0).sum(1).sum(1):
            if i == 0:
                count2.append(1)
            else:
                count2.append(0)
        count = sum(np.array(count1) * np.array(count2))
        # print(count)

        denominator = denominator.sum(0).sum(1).sum(1) + eps
        loss_per_channel = weights * (1 - (numerator / denominator)
                                      )  # Channel-wise weights
        # print(loss_per_channel) # 每一个类别的平均dice
        return (loss_per_channel.sum() - count) / (output.size(1) - count)
        # return loss_per_channel.sum() / output.size(1)


class CrossEntropy2D(nn.Module):
    """
    2D Cross-entropy loss implemented as negative log likelihood
    """
    def __init__(self, weight=None, reduction='none'):
        super(CrossEntropy2D, self).__init__()
        self.nll_loss = nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, inputs, targets):
        return self.nll_loss(inputs, targets)


class CombinedLoss(nn.Module):
    """
    For CrossEntropy the input has to be a long tensor
    Args:
        -- inputx N x C x H x W (其中N为batch_size)
        -- target - N x H x W - int type
        -- weight - N x H x W - float
    """
    def __init__(self, weight_dice, weight_ce):
        super(CombinedLoss, self).__init__()
        self.cross_entropy_loss = CrossEntropy2D()
        self.dice_loss = DiceLoss()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce

    def forward(self, inputx, target):
        target = target.type(torch.LongTensor)  # Typecast to long tensor
        if inputx.is_cuda:
            target = target.cuda()
        # print(inputx.min(),inputx.max())

        input_soft = F.softmax(inputx, dim=1)  # Along Class Dimension
        dice_val = torch.mean(self.dice_loss(input_soft, target))
        ce_val = torch.mean(self.cross_entropy_loss.forward(inputx, target))
        # ce_val = torch.mean(self.cross_entropy_loss.forward(inputx, target))
        total_loss = torch.add(torch.mul(dice_val, self.weight_dice),
                               torch.mul(ce_val, self.weight_ce))
        # print(weight.max())
        return total_loss, dice_val, ce_val
