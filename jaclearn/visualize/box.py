#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : box.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/03/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import six
import itertools
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Union, Optional

__all__ = ['add_bbox_patches', 'vis_bboxes']


def add_bbox_patches(ax, boxes: np.ndarray, class_name: Union[str, List[str]], add_text=True, legends: Optional[Union[str, List[str]]] = None):
    if legends is not None:
        assert len(legends) == len(boxes)
    else:
        legends = ['' for i in range(len(boxes))]

    if isinstance(class_name, six.string_types):
        class_name = itertools.repeat(class_name, len(boxes))

    for box, cls, leg in zip(boxes, class_name, legends):
        ax.add_patch(plt.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            fill=False, edgecolor='green', linewidth=3.5
        ))
        if add_text:
            ax.text(
                box[0], box[1] - 2,
                '{:s}{}'.format(cls, str(leg)),
                bbox=dict(facecolor='blue', alpha=0.5), fontsize=14, color='white'
            )
    return ax


def vis_bboxes(image: Image.Image, boxes: np.ndarray, class_name: Union[str, List[str]], add_text=True, legends: Optional[Union[str, List[str]]] = None, title: Optional[str] = None):
    title = title or "{} detection results".format(class_name)

    fig, ax = plt.subplots(figsize=(12, int(12 / image.width * image.height)))
    fig.tight_layout()
    ax.imshow(image, aspect='equal')
    add_bbox_patches(ax, boxes, class_name, add_text=add_text, legends=legends)
    ax.set_title(title, fontsize=14)
    ax.axis('off')

    return fig, ax

