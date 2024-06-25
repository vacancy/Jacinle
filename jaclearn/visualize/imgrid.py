#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : imgrid.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/19/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List


__all__ = ['image_grid', 'auto_image_grid_mplib']


def image_grid(all_images, grid_desc):
    """
    Create a image grid given the description. The description is a list of axis desc, of format: %d[h|v].
    If the first number n is a positive number, every n images will be concatenated horizontally or vertically.
    We allow exactly one axis desc to be only [h|v], meaning the number of images of that axis will be automatically
    inferred.

    Args:
        all_images: A list of images. Should be np.ndarray of shape (h, w, c).
        grid_desc: The grid description.

    Returns:
        A single big image created.
    """

    axes_info = []
    auto_infer_dim = len(all_images)
    for d in grid_desc:
        if d == 'h' or d == 'v':
            axes_info.append([None, d])
        else:
            assert d.endswith('h') or d.endswith('v'), d
            length, axis = int(d[:-1]), d[-1]
            assert auto_infer_dim % length == 0, 'Length of all_images should be divided by axes_info, ' \
                                                 'got {} and {}'.format(len(all_images), grid_desc)
            axes_info.append((length, axis))
            auto_infer_dim //= length

    for i, info in enumerate(axes_info):
        if info[0] is None:
            axes_info[i] = (auto_infer_dim, info[1])

    def stack(i, sequence):
        axis = axes_info[i][1]
        if len(sequence) == 1:
            return sequence[0]
        if axis == 'h':
            return np.hstack(sequence)
        return np.vstack(sequence)

    n = len(axes_info)

    def recursive_concat(i, sequence):
        if i == n - 1:
            return stack(i, sequence)

        nr_parts = axes_info[i][0]
        length = len(sequence) // nr_parts
        parts = []
        for j in range(nr_parts):
            part = recursive_concat(i+1, sequence[j*length:(j+1)*length])
            parts.append(part)
        return stack(i, parts)

    return recursive_concat(0, all_images)


def auto_image_grid_mplib(images: List[np.ndarray], images_title: Optional[List[str]] = None, global_title: Optional[str] = None, show: bool = True):
    """
    Automatically create a grid for the images.

    Args:
        images: a list of images. Should be np.ndarray of shape (h, w, c).
        images_title: the title for each image.
        global_title: the title for the whole image grid.
        show: whether to show the image grid using plt.show().

    Returns:
        the figure object.
    """

    n = len(images)

    if n in AUTO_IMAGE_GRID_DESC:
        nr_rows, nr_cols = AUTO_IMAGE_GRID_DESC[n]
    else:
        nr_cols = 5
        nr_rows = (n + nr_cols - 1) // nr_cols
    fig, axes = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*4, nr_rows*4))
    for i, image, title in zip(range(n), images, images_title if images_title is not None else [None] * n):
        ax = axes[i // nr_cols, i % nr_cols]
        ax.imshow(image)
        if title is not None:
            ax.set_title(title)
        ax.axis('off')

    if global_title is not None:
        # Use bold font for the global title
        fig.suptitle(global_title, fontweight='bold')
    fig.tight_layout()
    if show:
        plt.show()

    return fig


AUTO_IMAGE_GRID_DESC = {
    1: (1, 1),
    2: (1, 2),
    3: (1, 3),
    4: (2, 2),
    5: (1, 5),
    6: (2, 3),
    7: (2, 4),
    8: (2, 4),
    9: (2, 5),
    10: (2, 5),
}
