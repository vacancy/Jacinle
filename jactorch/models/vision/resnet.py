#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : resnet.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/31/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import math
import functools

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from jactorch.io import load_state_dict
from jactorch.nn import ResidualConvBlock, ResidualConvBottleneck


__all__ = ['ResNet', 'make_resnet',
           'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
           'reset_resnet_parameters']


class ResNet(nn.Module):
    def __init__(self, block, layers, incl_gap=False, num_classes=1000):
        super(ResNet, self).__init__()

        self.incl_gap = incl_gap
        self.incl_cls = self.incl_gap and num_classes is not None

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if self.incl_gap:
            self.avgpool = nn.AvgPool2d(7, stride=1)
        if self.incl_cls:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

    def reset_parameters(self):
        return reset_resnet_parameters(self)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.incl_gap:
            x = self.avgpool(x)

        if self.incl_cls:
            x = x.view(x.size(0), -1)
            x = self.fc(x)

        return x


cfgs = {
    'resnet18': (ResidualConvBlock, [2, 2, 2, 2]),
    'resnet34': (ResidualConvBlock, [3, 4, 6, 3]),
    'resnet50': (ResidualConvBottleneck, [3, 4, 6, 3]),
    'resnet101': (ResidualConvBottleneck, [3, 4, 23, 3]),
    'resnet152': (ResidualConvBottleneck, [3, 8, 36, 3]),
}

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def make_resnet(net_id, pretrained, incl_gap=True, num_classes=1000):
    model = ResNet(*cfgs[net_id], incl_gap=incl_gap, num_classes=num_classes)
    if pretrained:
        pretrained_model = model_zoo.load_url(model_urls[net_id])
        if num_classes != 1000:
            del pretrained_model['fc.weight']
            del pretrained_model['fc.bias']

        try:
            load_state_dict(model, pretrained_model)
        except KeyError:
            pass  # Intentionally ignore the key error.
    return model


def make_resnet_contructor(net_id):
    func = functools.partial(make_resnet, net_id=net_id)
    func.__name__ = net_id
    func.__doc__ = net_id.replace('resnet', 'ResNet-')
    return func


resnet18 = make_resnet_contructor('resnet18')
resnet34 = make_resnet_contructor('resnet34')
resnet50 = make_resnet_contructor('resnet50')
resnet101 = make_resnet_contructor('resnet101')
resnet152 = make_resnet_contructor('resnet152')


def reset_resnet_parameters(m, fc_std=0.01, bfc_std=0.001):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, fc_std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Bilinear):
        m.weight.data.normal_(0, bfc_std)
        if m.bias is not None:
            m.bias.data.zero_()
    else:
        for sub in m.modules():
            if m != sub:
                reset_resnet_parameters(sub, fc_std=fc_std, bfc_std=bfc_std)
