''' Incremental-Classifier Learning 
 Authors : Khurram Javed, Muhammad Talha Paracha
 Maintainer : Khurram Javed
 Lab : TUKL-SEECS R&D Lab
 Email : 14besekjaved@seecs.edu.pk '''

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from .res_utils import DownsampleA


class ResNetBasicblock(nn.Module):
    expansion = 1
    """
    RexNet basicblock (https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua)
    """

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResNetBasicblock, self).__init__()

        self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(planes)

        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.featureSize = 64

    def forward(self, x):
        residual = x

        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = F.relu(basicblock, inplace=True)

        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(x)

        return F.relu(residual + basicblock, inplace=True)


class CifarResNet(nn.Module):
    """
    ResNet optimized for the Cifar Dataset, as specified in
    https://arxiv.org/abs/1512.03385.pdf
    """

    def __init__(self, block, depth, num_classes, channels=3):
        """ Constructor
        Args:
          depth: number of layers.
          num_classes: number of classes
          base_width: base width
        """
        super(CifarResNet, self).__init__()

        self.featureSize = 64
        # Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        layer_blocks = (depth - 2) // 6
        self.dic = dict()

        self.num_classes = num_classes

        self.conv_1_3x3 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(16)

        self.inplanes = 16
        self.stage_1 = self._make_layer(block, 16, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 32, layer_blocks, 2)
        self.stage_3 = self._make_layer(block, 64, layer_blocks, 2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc0 = nn.Linear(64 * block.expansion, 10)
        self.fc1 = nn.Linear(64 * block.expansion, 10)
        self.fc2 = nn.Linear(64 * block.expansion, 10)
        self.fc3 = nn.Linear(64 * block.expansion, 10)
        self.fc4 = nn.Linear(64 * block.expansion, 10)
        self.fc5 = nn.Linear(64 * block.expansion, 10)
        self.fc6 = nn.Linear(64 * block.expansion, 10)
        self.fc7 = nn.Linear(64 * block.expansion, 10)
        self.fc8 = nn.Linear(64 * block.expansion, 10)
        self.fc9 = nn.Linear(64 * block.expansion, 10)
        self.fc_lst = [self.fc0, self.fc1, self.fc2, self.fc3, self.fc4, self.fc5,
                       self.fc6, self.fc7, self.fc8, self.fc9]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, fc_ind, feature=False, T=1, labels=False, per_logit=False,
                scale=None, keep=None, target=None, domain=False, logit=False, fea=False, sfm=False):
        loss = 0
        log = []
        sfm_lst = []
        # log = torch.zeros(len(x), self.num_classes, dtype=torch.float32).cuda()
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if fea:
            x = x.detach()
        if feature:
            return x / torch.norm(x, 2, 1).unsqueeze(1)
        if logit:
            for ind in range(fc_ind+1):
                pred = self.fc_lst[ind](x) / T
                # pred = self.fc_lst[ind](x)
                # pred = (pred - pred.max(dim=1)[0].unsqueeze(1)) / T
                # sfm = F.softmax(pred, dim=1)
                log.append(pred)
                # log += pred
            log = torch.cat(log, dim=1)
            return log
        if sfm:
            for ind in range(fc_ind+1):
                pred = self.fc_lst[ind](x) / T
                sfm = F.softmax(pred, dim=1)
                sfm_lst.append(sfm)
            sfm_lst = torch.cat(sfm_lst, dim=1)
            return sfm_lst
        x = self.fc_lst[fc_ind](x) / T
        if per_logit:
            return x
        # x = self.fc_lst[fc_ind](x)
        # x = (x - x.max(dim=1)[0].unsqueeze(1)) / T
        if keep is not None:
            x = x[:, keep[0]:keep[1]]
        if labels:
            return F.softmax(x, dim=1)

        if scale is not None:
            temp = F.softmax(x, dim=1)
            temp = temp * scale
            return temp

        return F.log_softmax(x, dim=1)

    def forwardFeature(self, x):
        pass


def resnet20(num_classes=10):
    """Constructs a ResNet-20 model for CIFAR-10 (by default)
    Args:
      num_classes (uint): number of classes
    """
    model = CifarResNet(ResNetBasicblock, 20, num_classes)
    return model


def resnet10mnist(num_classes=10):
    """Constructs a ResNet-20 model for CIFAR-10 (by default)
    Args:
      num_classes (uint): number of classes
    """
    model = CifarResNet(ResNetBasicblock, 10, num_classes, 1)
    return model


def resnet20mnist(num_classes=10):
    """Constructs a ResNet-20 model for CIFAR-10 (by default)
    Args:
      num_classes (uint): number of classes
    """
    model = CifarResNet(ResNetBasicblock, 20, num_classes, 1)
    return model


def resnet32mnist(num_classes=10, channels=1):
    model = CifarResNet(ResNetBasicblock, 32, num_classes, channels)
    return model


def resnet32(num_classes=10):
    """Constructs a ResNet-32 model for CIFAR-10 (by default)
    Args:
      num_classes (uint): number of classes
    """
    model = CifarResNet(ResNetBasicblock, 32, num_classes)
    return model


def resnet44(num_classes=10):
    """Constructs a ResNet-44 model for CIFAR-10 (by default)
    Args:
      num_classes (uint): number of classes
    """
    model = CifarResNet(ResNetBasicblock, 44, num_classes)
    return model


def resnet56(num_classes=10):
    """Constructs a ResNet-56 model for CIFAR-10 (by default)
    Args:
      num_classes (uint): number of classes
    """
    model = CifarResNet(ResNetBasicblock, 56, num_classes)
    return model


def resnet110(num_classes=10):
    """Constructs a ResNet-110 model for CIFAR-10 (by default)
    Args:
      num_classes (uint): number of classes
    """
    model = CifarResNet(ResNetBasicblock, 110, num_classes)
    return model
