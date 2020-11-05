#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : transforms.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/03/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import random

import torch
import torchvision.transforms as transforms

import jactorch.transforms.image as jac_transforms

from . import functional as F

__all__ = ["Compose", "Lambda", "ToTensor", "NormalizeBbox", "DenormalizeBbox", "Normalize", "Resize", "CenterCrop", "Pad",
           "RandomCrop", "RandomHorizontalFlip", "RandomVerticalFlip", "RandomResizedCrop",
           "LinearTransformation", "ColorJitter", "RandomRotation", "Grayscale", "RandomGrayscale",
           "PadMultipleOf"]


class Compose(transforms.Compose):
    def __call__(self, img, bbox):
        """
        Returns the image.

        Args:
            self: (todo): write your description
            img: (todo): write your description
            bbox: (array): write your description
        """
        for t in self.transforms:
            img, bbox = t(img, bbox)
        return img, bbox


class Lambda(transforms.Lambda):
    def __call__(self, img, bbox):
        """
        Calls the given image.

        Args:
            self: (todo): write your description
            img: (todo): write your description
            bbox: (array): write your description
        """
        return self.lambd(img, bbox)


class ToTensor(transforms.ToTensor):
    def __call__(self, img, bbox):
        """
        Calls bbox.

        Args:
            self: (todo): write your description
            img: (todo): write your description
            bbox: (array): write your description
        """
        # TODO(Jiayuan Mao @ 07/23): check whether bboxes are out of the image.
        return super().__call__(img), torch.from_numpy(bbox)


class NormalizeBbox(object):
    def __call__(self, img, bbox):
        """
        Normalize the given image.

        Args:
            self: (todo): write your description
            img: (todo): write your description
            bbox: (array): write your description
        """
        return F.normalize_bbox(img, bbox)


class DenormalizeBbox(object):
    def __call__(self, img, bbox):
        """
        Denormalize the image.

        Args:
            self: (todo): write your description
            img: (todo): write your description
            bbox: (array): write your description
        """
        return F.denormalize_bbox(img, bbox)


class Normalize(transforms.Normalize):
    def __call__(self, img, bbox):
        """
        Calls the call ().

        Args:
            self: (todo): write your description
            img: (todo): write your description
            bbox: (array): write your description
        """
        return super().__call__(img), bbox


class Resize(transforms.Resize):
    # Assuming bboxdinates are 0/1-normalized.
    def __call__(self, img, bbox):
        """
        Calls the call ().

        Args:
            self: (todo): write your description
            img: (todo): write your description
            bbox: (array): write your description
        """
        return super().__call__(img), bbox


class CenterCrop(transforms.CenterCrop):
    def __call__(self, img, bbox):
        """
        Crop a crop on the image.

        Args:
            self: (todo): write your description
            img: (todo): write your description
            bbox: (array): write your description
        """
        return F.center_crop(img, bbox, self.size)


class Pad(transforms.Pad):
    def __call__(self, img, bbox):
        """
        Pushes the image to the image.

        Args:
            self: (todo): write your description
            img: (todo): write your description
            bbox: (array): write your description
        """
        return F.pad(img, bbox, self.padding, self.fill)


class RandomCrop(transforms.RandomCrop):
    def __call__(self, img, bbox):
        """
        Pushes the image.

        Args:
            self: (todo): write your description
            img: (todo): write your description
            bbox: (array): write your description
        """
        if self.padding > 0:
            img = F.pad(img, bbox, self.padding)
        i, j, h, w = self.get_params(img, self.size)
        return F.crop(img, bbox, i, j, h, w)


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __call__(self, img, bbox):
        """
        Wrapper on - > bbox

        Args:
            self: (todo): write your description
            img: (todo): write your description
            bbox: (array): write your description
        """
        if random.random() < 0.5:
            return F.hflip(img, bbox)
        return img, bbox


class RandomVerticalFlip(transforms.RandomVerticalFlip):
    def __call__(self, img, bbox):
        """
        Returns a bbox from the given image.

        Args:
            self: (todo): write your description
            img: (todo): write your description
            bbox: (array): write your description
        """
        if random.random() < 0.5:
            return F.vflip(img, bbox)
        return img, bbox


class RandomResizedCrop(transforms.RandomResizedCrop):
    def __call__(self, img, bbox):
        """
        Interpolation.

        Args:
            self: (todo): write your description
            img: (todo): write your description
            bbox: (array): write your description
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, bbox, i, j, h, w, self.size, self.interpolation)


class Grayscale(transforms.Grayscale):
    def __call__(self, img, bbox):
        """
        Calls the call ().

        Args:
            self: (todo): write your description
            img: (todo): write your description
            bbox: (array): write your description
        """
        return super().__call__(img), bbox


class RandomGrayscale(transforms.RandomGrayscale):
    def __call__(self, img, bbox):
        """
        Calls the call ().

        Args:
            self: (todo): write your description
            img: (todo): write your description
            bbox: (array): write your description
        """
        return super().__call__(img), bbox


class LinearTransformation(transforms.LinearTransformation):
    def __call__(self, tensor, bbox):
        """
        Return the bbox.

        Args:
            self: (todo): write your description
            tensor: (todo): write your description
            bbox: (array): write your description
        """
        return super().__call__(tensor), bbox


class ColorJitter(transforms.ColorJitter):
    def __call__(self, img, bbox):
        """
        Calls the call ().

        Args:
            self: (todo): write your description
            img: (todo): write your description
            bbox: (array): write your description
        """
        return super().__call__(img), bbox


class RandomRotation(transforms.RandomRotation):
    def __call__(self, img, bbox):
        """
        Returns the image.

        Args:
            self: (todo): write your description
            img: (todo): write your description
            bbox: (array): write your description
        """
        assert self.degrees[0] == self.degrees[1] == 0
        angle = self.get_params(self.degrees)
        return F.rotate(img, bbox, angle, self.resample, self.expand, self.center)


class PadMultipleOf(jac_transforms.PadMultipleOf):
    def __call__(self, img, coor):
        """
        Return the image ator.

        Args:
            self: (todo): write your description
            img: (todo): write your description
            coor: (array): write your description
        """
        return F.pad_multiple_of(img, coor, self.multiple)

