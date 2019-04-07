#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : transforms.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/18/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import random
import numbers
import collections

from PIL import Image

import torch
import torchvision.transforms.transforms as torch_transforms

from jacinle.utils.argument import get_2dshape
from jacinle.utils.enum import JacEnum
from jacinle.utils.defaults import defaults_manager

from .functional import image as fimage
from .functional import coor as fcoor
from .functional import bbox as fbbox
from .functional._utils import get_rotation_matrix, get_size_multiple_of

__all__ = [
    "TransformDataTypes", "TransformGuide", "TransformBase", "TransformFunctionBase", "TransformFunctionBaseImageOnly",
    "Compose", "Lambda", "RandomApply", "RandomOrder", "RandomChoice",
    "ToTensor", "ToPILImage", "Normalize", "NormalizeCoordinates", "DenormalizeCoordinates",
    "Crop", "CenterCrop", "RandomCrop", "Pad", "PadMultipleOf",
    "HFlip", "VFlip", "RandomHorizontalFlip", "RandomVerticalFlip",
    "Resize", "ResizeMultipleOf", "RandomResizedCrop",
    "FiveCrop", "TenCrop",
    "Rotate", "RandomRotation",
    "LinearTransformation", "ColorJitter", "RandomRotation", "Grayscale", "RandomGrayscale",
]


class TransformDataTypes(JacEnum):
    IMAGE = 'image'
    COOR = 'coor'
    BBOX = 'bbox'


class TransformGuide(object):
    def __init__(self, transform_guide):
        self.transform_guide = transform_guide

    def keys(self):
        return self.transform_guide.keys()

    def items(self):
        return self.transform_guide.items()

    def gen(self, feed_dict):
        for k, v in self.transform_guide.items():
            if k in feed_dict:
                yield k, feed_dict[k], TransformDataTypes.from_string(v['type']), v.get('dep', [])

    @defaults_manager.wrap_custom_as_default(is_local=True)
    def as_default(self):
        yield self


default_transform_guide = TransformGuide({
    'image': {'type': 'image'},
    'coor': {'type': 'coor', 'dep': ['image']},
    'bbox': {'type': 'bbox', 'dep': ['image']}
})
get_default_transform_guide = defaults_manager.gen_get_default(TransformGuide, lambda: default_transform_guide)


class TransformBase(object):
    def __init__(self, tg=None):
        self.transform_guide = tg
        if self.transform_guide is None:
            self.transform_guide = get_default_transform_guide()

    def _get_image(self, feed_dict):
        for k, v, type, dep in self.transform_guide.gen(feed_dict):
            if type is TransformDataTypes.IMAGE:
                return v
        return None

    def ezcall(self, image=None, coor=None, bbox=None):
        feed_dict = dict()
        for k in default_transform_guide.keys():
            if locals()[k] is not None:
                feed_dict[k] = locals()[k]
        feed_dict = self(feed_dict)

        def ret():
            for k in default_transform_guide.keys():
                if k in feed_dict:
                    yield feed_dict[k]
        ret = tuple(ret())

        if len(ret) == 1:
            return ret[0]
        return ret

    def __call__(self, feed_dict=None, **kwargs):
        if feed_dict is not None and not isinstance(feed_dict, collections.Mapping):
            return self.ezcall(feed_dict, **kwargs)

        feed_dict = feed_dict or {}
        feed_dict.update(**kwargs)
        feed_dict = self.call_feed_dict(feed_dict)
        return feed_dict

    def call_feed_dict(self, feed_dict):
        raise NotImplementedError()


class TransformFunctionBase(TransformBase):
    def call_feed_dict(self, feed_dict):
        output_dict = feed_dict.copy()
        for k, v, type, dep in self.transform_guide.gen(feed_dict):
            if type is TransformDataTypes.IMAGE:
                output_dict[k] = self.call_image(v)
            elif type is TransformDataTypes.COOR:
                assert len(dep) == 1 and dep[0] in feed_dict, 'Invalid dependency for {}: {}.'.format(k, dep)
                output_dict[k] = self.call_coor(feed_dict[dep[0]], v)
            elif type is TransformDataTypes.BBOX:
                assert len(dep) == 1 and dep[0] in feed_dict, 'Invalid dependency for {}: {}.'.format(k, dep)
                output_dict[k] = self.call_bbox(feed_dict[dep[0]], v)
        return output_dict

    def call_image(self, img):
        raise NotImplementedError('Unsupported transform {} for data type "image".'.format(self.__class__.__name__))

    def call_coor(self, img, coor):
        raise NotImplementedError('Unsupported transform {} for data type "coor".'.format(self.__class__.__name__))

    def call_bbox(self, img, bbox):
        raise NotImplementedError('Unsupported transform {} for data type "bbox".'.format(self.__class__.__name__))


class TransformFunctionBaseImageOnly(TransformFunctionBase):
    def call_coor(self, img, coor):
        return coor

    def call_bbox(self, img, bbox):
        return bbox


class Compose(torch_transforms.Compose):
    def __call__(self, feed_dict=None, **kwargs):
        if feed_dict is not None and not isinstance(feed_dict, collections.Mapping):
            return self.ezcall(feed_dict, **kwargs)

        feed_dict = feed_dict or {}
        feed_dict.update(**kwargs)
        feed_dict = super().__call__(feed_dict)
        return feed_dict

    ezcall = TransformBase.ezcall


class RandomApply(torch_transforms.RandomApply):
    def __call__(self, feed_dict=None, **kwargs):
        if feed_dict is not None and not isinstance(feed_dict, collections.Mapping):
            return self.ezcall(feed_dict, **kwargs)

        feed_dict = feed_dict or {}
        feed_dict.update(**kwargs)
        feed_dict = super().__call__(feed_dict)
        return feed_dict

    ezcall = TransformBase.ezcall


class RandomOrder(torch_transforms.RandomOrder):
    def __call__(self, feed_dict=None, **kwargs):
        if feed_dict is not None and not isinstance(feed_dict, collections.Mapping):
            return self.ezcall(feed_dict, **kwargs)

        feed_dict = feed_dict or {}
        feed_dict.update(**kwargs)
        feed_dict = super().__call__(feed_dict)
        return feed_dict

    ezcall = TransformBase.ezcall


class RandomChoice(torch_transforms.RandomChoice):
    def __call__(self, feed_dict=None, **kwargs):
        if feed_dict is not None and not isinstance(feed_dict, collections.Mapping):
            return self.ezcall(feed_dict, **kwargs)

        feed_dict = feed_dict or {}
        feed_dict.update(**kwargs)
        feed_dict = super().__call__(feed_dict)
        return feed_dict


class Lambda(torch_transforms.Lambda):
    def __call__(self, feed_dict=None, **kwargs):
        if feed_dict is not None and not isinstance(feed_dict, collections.Mapping):
            return self.ezcall(feed_dict, **kwargs)

        feed_dict = feed_dict or {}
        feed_dict.update(**kwargs)
        feed_dict = super().__call__(feed_dict)
        return feed_dict

    ezcall = TransformBase.ezcall


class ToTensor(TransformFunctionBase):
    def call_image(self, img):
        return fimage.to_tensor(img)

    def call_coor(self, img, coor):
        coor = fcoor.refresh_valid(img, coor)
        return torch.tensor(coor)

    def call_bbox(self, img, bbox):
        bbox = fbbox.refresh_valid(img, bbox)
        return torch.tensor(bbox)

    __doc__ = torch_transforms.ToTensor.__doc__
    __repr__ = torch_transforms.ToTensor.__repr__


class ToPILImage(TransformFunctionBaseImageOnly):
    def __init__(self, mode=None, tg=None):
        super().__init__(tg)
        self.mode = mode

    def call_image(self, img):
        return fimage.to_pil_image(img, self.mode)

    __doc__ = torch_transforms.ToPILImage.__doc__
    __repr__ = torch_transforms.ToPILImage.__repr__


class Normalize(TransformFunctionBaseImageOnly):
    def __init__(self, mean, std, tg=None):
        super().__init__(tg)
        self.mean = mean
        self.std = std

    def call_image(self, tensor):
        return fimage.normalize(tensor, self.mean, self.std)

    __doc__ = torch_transforms.Normalize.__doc__
    __repr__ = torch_transforms.Normalize.__repr__


class NormalizeCoordinates(TransformFunctionBase):
    def call_image(self, img):
        return img

    def call_coor(self, img, coor):
        return fcoor.normalize_coor(img, coor)

    def call_bbox(self, img, bbox):
        return fbbox.normalize_bbox(img, bbox)


class DenormalizeCoordinates(TransformFunctionBase):
    def call_image(self, img):
        return img

    def call_coor(self, img, coor):
        return fcoor.denormalize_coor(img, coor)

    def call_bbox(self, img, bbox):
        return fbbox.denormalize_box(img, bbox)


class Crop(TransformFunctionBase):
    def __init__(self, x, y, w, h, tg=None):
        super().__init__(tg)
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def call_image(self, img):
        return fimage.crop(img, self.x, self.y, self.w, self.h)

    def call_coor(self, img, coor):
        return fcoor.crop(coor, self.x, self.y, self.w, self.h)

    def call_bbox(self, img, bbox):
        return fbbox.crop(bbox, self.x, self.y, self.w, self.h)


class CenterCrop(TransformBase):
    def __init__(self, size, tg=None):
        super().__init__(tg)
        self.size = get_2dshape(size)

    def call_feed_dict(self, feed_dict):
        img = self._get_image(feed_dict)
        w, h = img.size
        tw, th = self.size
        x = int(round((w - tw) / 2.))
        y = int(round((h - th) / 2.))
        return Crop(x, y, tw, th, tg=self.transform_guide)(feed_dict)

    __doc__ = torch_transforms.CenterCrop.__doc__
    __repr__ = torch_transforms.CenterCrop.__repr__


class RandomCrop(TransformBase):
    def __init__(self, size, padding=0, pad_if_needed=False, tg=None):
        super().__init__(tg)
        self.size = get_2dshape(size)
        self.padding = padding
        self.pad_if_needed = pad_if_needed

    def call_feed_dict(self, feed_dict):
        if self.padding > 0:
            feed_dict = Pad(self.padding, tg=self.transform_guide)(feed_dict)

        img = self._get_image(feed_dict)
        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            feed_dict = Pad((int((1 + self.size[1] - img.size[0]) / 2), 0), tg=self.transform_guide)(feed_dict)
        img = self._get_image(feed_dict)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            feed_dict = Pad((0, int((1 + self.size[0] - img.size[1]) / 2)), tg=self.transform_guide)(feed_dict)

        i, j, h, w = torch_transforms.RandomCrop.get_params(img, self.size)
        return Crop(j, i, w, h, tg=self.transform_guide)(feed_dict)

    __doc__ = torch_transforms.RandomCrop.__doc__
    __repr__ = torch_transforms.RandomCrop.__repr__


class Pad(TransformFunctionBase):
    def __init__(self, padding, mode='constant', fill=0, tg=None):
        super().__init__(tg)

        assert isinstance(padding, (numbers.Number, tuple))
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert mode in ['constant', 'edge', 'reflect', 'symmetric']
        if isinstance(padding, collections.Sequence) and len(padding) not in [2, 4]:
            raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                             "{} element tuple".format(len(padding)))

        self.padding = padding
        self.fill = fill
        self.padding_mode = mode

    def call_image(self, img):
        return fimage.pad(img, self.padding, self.padding_mode, self.fill)

    def call_coor(self, img, coor):
        return fcoor.pad(coor, self.padding)

    def call_bbox(self, img, bbox):
        return fbbox.pad(bbox, self.padding)

    __doc__ = torch_transforms.Pad.__doc__
    __repr__ = torch_transforms.Pad.__repr__


class PadMultipleOf(TransformBase):
    def __init__(self, multiple, residual=0, mode='constant', fill=0, tg=None):
        super().__init__(tg)
        self.multiple = multiple
        self.residual = residual
        self.mode = mode
        self.fill = fill

    def call_feed_dict(self, feed_dict):
        img = self._get_image(feed_dict)
        h, w = img.height, img.width
        hh, ww = get_size_multiple_of(h, w, self.multiple, self.residual)
        if h != hh or w != ww:
            feed_dict = Pad((0, 0, ww - w, hh - h), mode=self.mode, fill=self.fill, tg=self.transform_guide)(feed_dict)
        return feed_dict


class HFlip(TransformFunctionBase):
    def call_image(self, img):
        return fimage.hflip(img)

    def call_coor(self, img, coor):
        return fcoor.hflip(img, coor)

    def call_bbox(self, img, bbox):
        return fbbox.hflip(img, bbox)


class VFlip(TransformFunctionBase):
    def call_image(self, img):
        return fimage.vflip(img)

    def call_coor(self, img, coor):
        return fcoor.vflip(img, coor)

    def call_bbox(self, img, bbox):
        return fbbox.vflip(img, bbox)


class RandomHorizontalFlip(TransformBase):
    def __init__(self, p=0.5, tg=None):
        super().__init__(tg)
        self.p = p

    def call_feed_dict(self, feed_dict):
        if random.random() < self.p:
            return HFlip(tg=self.transform_guide)(feed_dict)
        return feed_dict

    __doc__ = torch_transforms.RandomHorizontalFlip.__doc__
    __repr__ = torch_transforms.RandomHorizontalFlip.__repr__


class RandomVerticalFlip(TransformBase):
    def __init__(self, p=0.5, tg=None):
        super().__init__(tg)
        self.p = p

    def call_feed_dict(self, feed_dict):
        if random.random() < self.p:
            return VFlip(tg=self.transform_guide)(feed_dict)
        return feed_dict

    __doc__ = torch_transforms.RandomVerticalFlip.__doc__
    __repr__ = torch_transforms.RandomVerticalFlip.__repr__


class Resize(TransformFunctionBase):
    def __init__(self, size, interpolation=Image.BILINEAR, tg=None):
        super().__init__(tg)
        self.size = get_2dshape(size)
        self.interpolation = interpolation

    def call_image(self, img):
        return fimage.resize(img, self.size, self.interpolation)

    def call_coor(self, img, coor):
        return fcoor.resize(img, coor, self.size)

    def call_bbox(self, img, bbox):
        return fbbox.resize(img, bbox, self.size)

    __doc__ = torch_transforms.Resize.__doc__
    __repr__ = torch_transforms.Resize.__repr__


class ResizeMultipleOf(TransformBase):
    def __init__(self, multiple, residual=0, interpolation=Image.NEAREST, tg=None):
        super().__init__(tg)
        self.multiple = multiple
        self.residual = residual
        self.interpolation = interpolation

    def call_feed_dict(self, feed_dict):
        img = self._get_image(feed_dict)
        h, w = img.height, img.width
        hh, ww = get_size_multiple_of(h, w, self.multiple, self.residual)
        if h != hh or w != ww:
            feed_dict = Resize((hh, ww), interpolation=self.interpolation, tg=self.transform_guide)(feed_dict)
        return feed_dict


class RandomResizedCrop(TransformBase):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR, tg=None):
        super().__init__(tg)
        self.size = (size, size)
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    def call_feed_dict(self, feed_dict):
        img = self._get_image(feed_dict)
        i, j, h, w = torch_transforms.RandomResizedCrop.get_params(img, self.scale, self.ratio)
        feed_dict = Crop(j, i, w, h, tg=self.transform_guide)(feed_dict)
        feed_dict = Resize(self.size, self.interpolation, tg=self.transform_guide)(feed_dict)
        return feed_dict

    __doc__ = torch_transforms.RandomResizedCrop.__doc__
    __repr__ = torch_transforms.RandomResizedCrop.__repr__


class FiveCrop(TransformFunctionBase):
    def __init__(self, size, tg=None):
        super().__init__(tg)
        self.size = get_2dshape(size)

    def call_image(self, img):
        return fimage.five_crop(img, self.size)

    __doc__ = torch_transforms.FiveCrop.__doc__
    __repr__ = torch_transforms.FiveCrop.__repr__


class TenCrop(TransformFunctionBase):
    def __init__(self, size, tg=None):
        super().__init__(tg)
        self.size = get_2dshape(size)

    def call_image(self, img):
        return fimage.ten_crop(img, self.size)

    __doc__ = torch_transforms.TenCrop.__doc__
    __repr__ = torch_transforms.TenCrop.__repr__


class _AffineHelper(TransformFunctionBase):
    def __init__(self, owner, matrix, tg):
        super().__init__(tg)
        self.owner = owner
        self.matrix = matrix

    def call_image(self, img):
        return img.rotate(
            self.owner.angle,
            resample=self.owner.resample, expand=self.owner.expand,
            center=self.owner.center, translate=self.owner.translate
        )

    def call_coor(self, img, coor):
        return fcoor.affine(coor, self.matrix)

    def call_bbox(self, img, bbox):
        return fbbox.affine(bbox, self.matrix)


class Rotate(TransformBase):
    def __init__(self, angle, resample=False, crop=False, expand=False, center=None, translate=None, tg=None):
        super().__init__(tg)
        self.angle = angle
        self.resample = resample
        self.crop = crop
        self.expand = expand
        self.center = center
        self.translate = translate

    def call_feed_dict(self, feed_dict):
        img = self._get_image(feed_dict)
        matrix, extra_crop = get_rotation_matrix(img, self.angle, self.crop, self.expand, self.center, self.translate)
        feed_dict = _AffineHelper(self, matrix, tg=self.transform_guide)(feed_dict)
        if extra_crop is not None:
            feed_dict = Crop(*extra_crop, tg=self.transform_guide)(feed_dict)
        return feed_dict


class RandomRotation(TransformBase):
    def __init__(self, degrees, resample=False, crop=False, expand=False, center=None, translate=None, tg=None):
        super().__init__(tg)
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.crop = crop
        self.resample = resample
        self.expand = expand
        self.center = center
        self.translate = translate

    def call_feed_dict(self, feed_dict):
        angle = torch_transforms.RandomRotation.get_params(self.degrees)
        return Rotate(angle, self.resample, self.crop, self.expand, self.center, self.translate, tg=self.transform_guide)(feed_dict)

    __doc__ = torch_transforms.RandomRotation.__doc__

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', crop={0}'.format(self.crop)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        if self.translate is not None:
            format_string += ', translate={0}'.format(self.translate)
        format_string += ')'
        return format_string


class LinearTransformation(TransformFunctionBaseImageOnly):
    def __init__(self, transformation_matrix, tg=None):
        super().__init__(tg)
        if transformation_matrix.size(0) != transformation_matrix.size(1):
            raise ValueError("transformation_matrix should be square. Got " +
                             "[{} x {}] rectangular matrix.".format(*transformation_matrix.size()))
        self.transformation_matrix = transformation_matrix

    def call_image(self, tensor):
        return torch_transforms.LinearTransformation(self.transformation_matrix)(tensor)

    __doc__ = torch_transforms.LinearTransformation.__doc__
    __repr__ = torch_transforms.LinearTransformation.__repr__


class ColorJitter(TransformFunctionBaseImageOnly):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, tg=None):
        super().__init__(tg)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def call_image(self, img):
        return torch_transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)(img)


class Grayscale(TransformFunctionBaseImageOnly):
    def __init__(self, num_output_channels=1, tg=None):
        super().__init__(tg)
        self.num_output_channels = num_output_channels

    def call_image(self, img):
        return fimage.to_grayscale(img, num_output_channels=self.num_output_channels)

    __doc__ = torch_transforms.Grayscale.__doc__
    __repr__ = torch_transforms.Grayscale.__repr__


class RandomGrayscale(TransformFunctionBaseImageOnly):
    def __init__(self, p=0.1, tg=None):
        super().__init__(tg)
        self.p = p

    def call_image(self, img):
        num_output_channels = 1 if img.mode == 'L' else 3
        if random.random() < self.p:
            return fimage.to_grayscale(img, num_output_channels=num_output_channels)
        return img

    __doc__ = torch_transforms.RandomGrayscale.__doc__
    __repr__ = torch_transforms.RandomGrayscale.__repr__
