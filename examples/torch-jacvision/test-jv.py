#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test-jv.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/04/2019
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import os.path as osp
from PIL import Image

import torch
import torchvision.transforms.functional as TF

import jactorch.vision as jv


def imread(path):
    return TF.to_tensor(Image.open(path).convert('L'))


def normabs(tensor):
    tensor = tensor.abs()
    tensor -= tensor.min()
    tensor /= tensor.max()
    return tensor


def imwrite(path, tensor):
    TF.to_pil_image(tensor).save(path)


def main():
    image = 1 - imread('./i_image.png')

    # image = torch.zeros((1, 10, 10), dtype=torch.float32)
    # image[0, 3:7, 3:7] = 1
    imwrite(osp.join('viz', 'original.png'), image)

    for op_name in [
            'normalized_box_smooth', 'gaussian_smooth',
            'erode', 'dilate', 'open', 'open', 'close', 'morph_grad', 'top_hat', 'black_hat',
            'image_gradient', 'sobel', 'scharr', 'laplacian'
    ]:
        if op_name == 'normalized_box_smooth':
            output = getattr(jv, op_name)(image, 5)
        elif op_name == 'gaussian_smooth':
            output = getattr(jv, op_name)(image, 15, 3)
        elif op_name in ['erode', 'dilate', 'open', 'open', 'close', 'morph_grad', 'top_hat', 'black_hat']:
            output = getattr(jv, op_name)(image, 3)
        elif op_name in ['image_gradient', 'sobel', 'scharr', 'laplacian']:
            output = normabs(getattr(jv, op_name)(image))

        imwrite(osp.join('viz', op_name + '.png'), output)

    for op in dir(jv):
        if op.islower() and op not in [
            'normalized_box_smooth', 'gaussian_smooth',
            'erode', 'dilate', 'open', 'open', 'close', 'morph_grad', 'top_hat', 'black_hat',
            'image_gradient', 'sobel', 'scharr', 'laplacian'
        ]:
            print(op)


if __name__ == '__main__':
    main()
