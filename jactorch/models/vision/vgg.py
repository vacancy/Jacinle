#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : vgg.py
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


__all__ = [
    'VGG', 'make_vgg',
    'vgg11', 'vgg11_bn',
    'vgg13', 'vgg13_bn',
    'vgg16', 'vgg16_bn',
    'vgg19', 'vgg19_bn',
    'reset_vgg_parameters'
]


class VGG(nn.Module):
    def __init__(self, cfg, batch_norm=False, incl_p5=True, incl_fcs=True, num_classes=1000):
        super(VGG, self).__init__()

        self.incl_p5 = incl_p5
        self.incl_fcs = self.incl_p5 and incl_fcs
        self.incl_cls = self.incl_fcs and num_classes is not None

        if not self.incl_p5 and cfg[-1] == 'M':
            cfg.pop()
        self.features = self.make_layers(cfg, batch_norm=batch_norm)

        if self.incl_fcs:
            self.fc = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
            )

        if self.incl_cls:
            self.classifier = nn.Linear(4096, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        reset_vgg_parameters(self)

    def forward(self, x):
        x = self.features(x)

        if self.incl_fcs:
            x = x.view(x.size(0), -1)
            x = self.fc(x)

        if self.incl_cls:
            x = self.classifier(x)

        return x

    @staticmethod
    def make_layers(cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

fc_mapping = {
    'fc.0.weight': 'classifier.0.weight',
    'fc.0.bias': 'classifier.0.bias',
    'fc.3.weight': 'classifier.3.weight',
    'fc.3.bias': 'classifier.3.bias',
    'classifier.weight': 'classifier.6.weight',
    'classifier.bias': 'classifier.6.bias'
}


def make_vgg(cfg_id, batch_norm, pretrained, url_id, incl_fcs=True, num_classes=1000):
    model = VGG(cfgs[cfg_id], batch_norm=batch_norm, incl_fcs=incl_fcs, num_classes=num_classes)
    if pretrained:
        pretrained_model = model_zoo.load_url(model_urls[url_id])
        for k, v in fc_mapping.items():
            pretrained_model[k] = pretrained_model.pop(v)
        if num_classes != 1000:
            del pretrained_model['classifier.weight']
            del pretrained_model['classifier.bias']

        try:
            load_state_dict(model, pretrained_model)
        except KeyError:
            pass  # Intentionally ignore the key error.
    return model


def make_vgg_contructor(cfg_id, url_id, batch_norm=False):
    func = functools.partial(make_vgg, cfg_id=cfg_id, url_id=url_id, batch_norm=batch_norm)
    func.__name__ = url_id
    func.__doc__ = url_id.replace('vgg', 'VGG ').replace('_bn', ' (with batch normalization)')
    func.__doc__ += '(configuration: {})'.format(cfg_id)
    return func


vgg11 = make_vgg_contructor('A', 'vgg11')
vgg13 = make_vgg_contructor('B', 'vgg13')
vgg16 = make_vgg_contructor('D', 'vgg16')
vgg19 = make_vgg_contructor('E', 'vgg19')
vgg11_bn = make_vgg_contructor('A', 'vgg11_bn', batch_norm=True)
vgg13_bn = make_vgg_contructor('B', 'vgg13_bn', batch_norm=True)
vgg16_bn = make_vgg_contructor('D', 'vgg16_bn', batch_norm=True)
vgg19_bn = make_vgg_contructor('E', 'vgg19_bn', batch_norm=True)


def reset_vgg_parameters(m, fc_std=0.01, bfc_std=0.001):
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
                reset_vgg_parameters(sub, fc_std=fc_std, bfc_std=bfc_std)
