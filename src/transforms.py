#!/usr/bin/env python
# coding=utf-8
"""
Data does not always come in its final processed form that is required for
training machine learning algorithms. We use transforms to perform some
manipulation of the data and make it suitable for training.

All TorchVision datasets have two parameters - `transform` to modify the
features and `target_transform` to modify the labels - that accept callables
containing the transformation logic. The `torchvision.transforms` module offers
several commonly-used transforms out of the box.

Transforms can be chained together using `torchvision.Transforms.Compose`. The
transformations that accept tensor images also accept batches of tensor images.
A Tensor Image is a tensor with (C, H, W) shape, where C is a number of
channels, H and W are image height and width. A batch of Tensor Images is a
tensor of (B, C, H, W) shape, where B is a number of images in the batch.

The expected range of the values of a tensor image is implicitely defined by
the tensor dtype. Tensor images with a float dtype are expected to have values
in [0, 1). Tensor images with an integer dtype are expected to have values in
[0, MAX_DTYPE] where MAX_DTYPE is the largest value that can be represented in
that dtype.
"""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torchvision.transforms as T  # type: ignore
from PIL import Image
from .configuration import BaseConfig

# BICUBIC = InterpolationMode.BICUBIC
BICUBIC = Image.BICUBIC

__all__ = ["TransformFactory", "BaseTransform", "print_size_warning"]

AUGMENTATIONS = [
    "grayscale",
    "fixsize",
    "resize",
    "scale_width",
    "scale_shortside",
    "zoom",
    "crop",
    "patch",
    "trim",
    "flip",
    "convert",
    "make_power_2",
]


class TransformFactory:
    """Augmentation factory"""
    def __init__(self) -> None:
        self._cfg = BaseConfig()

    @property
    def stage(self):
        return self._cfg.stage

    @stage.setter
    def stage(self, value):
        self._cfg.stage = value

    @property
    def cfg(self):
        """Get cfg dictionary"""
        return self._cfg

    @cfg.setter
    def cfg(self, cfg_obj: BaseConfig):
        self._cfg = cfg_obj

    @property
    def augmentations(self):
        return self._cfg.get("augmentations")

    @property
    def out_ch(self):
        return self._cfg.get("out_ch")

    @property
    def interp(self):
        return self._cfg.get("interp")

    @property
    def out_size(self):
        return self._cfg.get("out_size")

    @property
    def load_size(self):
        return self._cfg.get("load_size")

    @property
    def dataset_name(self):
        return self._cfg.get("dataset_name")

    @property
    def max_size(self):
        return self._cfg.get("max_size")

    @property
    def shortside_size(self):
        return self._cfg.get("shortside_size")

    @property
    def zoom_factor(self):
        return self._cfg.get("zoom_factor")

    @property
    def crop_size(self):
        return self._cfg.get("crop_size")

    @property
    def crop_pos(self):
        return self._cfg.get("crop_pos")

    @property
    def patch_size(self):
        return self._cfg.get("patch_size")

    @property
    def patch_stride(self):
        return self._cfg.get("patch_stride")

    @property
    def trim_size(self):
        return self._cfg.get("trim_size")

    @property
    def flip(self):
        return self._cfg.get("flip")

    @property
    def power_base(self):
        return self._cfg.get("power_base")

    def get_transform(self, stage: str = "train"):
        """Get T.Compose with the configured augmentations"""
        self.stage = stage

        transform = {}

        interp = self.interp
        if "grayscale" in self.augmentations:
            transform["grayscale"] = T.Grayscale(self.out_ch)

        if "fixsize" in self.augmentations:
            out_size = self.out_size
            transform["fixsize"] = T.Resize(out_size, interpolation=interp)

        if "resize" in self.augmentations:
            load_size = self.load_size
            if self.dataset_name == "gta2cityscapes":
                load_size[0] = load_size[0] // 2
            transform["resize"] = T.Resize(load_size, interpolation=interp)

        elif "scale_width" in self.augmentations:
            width, _ = self.out_size
            _, max_height = self.max_size
            lambd = lambda img: self.scale_width(
                img, width, max_height, interp=interp)
            transform["scale_width"] = T.Lambda(lambd)

        elif "scale_shortside" in self.augmentations:
            size = self.shortside_size
            lambd = lambda img: self.scale_shortside(img, size, interp=interp)
            transform["scale_shortside"] = T.Lambda(lambd)

        if "zoom" in self.augmentations:
            max_size = self.max_size
            factor = self.zoom_factor
            lambd = lambda img: self.random_zoom(
                img, max_size, factor=factor, interp=interp)
            transform["zoom"] = T.Lambda(lambd)

        if "crop" in self.augmentations:
            size = self.crop_size
            pos = self.crop_pos
            if pos is None:
                transform["crop"] = T.RandomCrop(size=size)
            else:
                lambd = lambda img: self.crop(img, size=size, pos=pos)
                transform["crop"] = T.Lambda(lambd)

        if "patch" in self.augmentations:
            size = self.patch_size
            stride = self.patch_stride
            lambd = lambda img: self.patch(img, size, stride=stride)
            transform["patch"] = T.Lambda(lambd)

        if "trim" in self.augmentations:
            lambd = lambda img: self.trim(img, self.trim_size)
            transform["trim"] = T.Lambda(lambd)

        if "flip" in self.augmentations:
            if self.flip is None:
                transform["flip"] = T.RandomHorizontalFlip()
            else:
                lambd = lambda img: self.flip_img(img, flip=self.flip)
                transform["flip"] = T.Lambda(lambd)

        if "convert" in self.augmentations:
            grayscale = bool("grayscale" in self.augmentations)
            lambd = lambda img: self.convert(img, grayscale=grayscale)
            transform["convert"] = T.Lambda(lambd)

        # Make power of 'base'
        base = self.power_base
        lambd = lambda img: self.make_power_2(img, base=base, interp=interp)
        transform["make_power_2"] = T.Lambda(lambd)

        transform_list = []
        for aug in self.augmentations:
            if isinstance(transform[aug], list):
                transform_list.extend(transform[aug])
            else:
                transform_list.append(transform[aug])

        transform_list.append(T.ToTensor())
        return T.Compose(transform_list)

    @staticmethod
    def scale_width(img, width, max_height, interp=BICUBIC):
        """Scale the given image's width."""
        augment = ScaleWidthTransform(width, max_height, interp=interp)
        return augment(img)

    @staticmethod
    def scale_shortside(img, size, interp=BICUBIC):
        """Scale the given image's shorter side."""
        augment = ScaleShortsideTransform(size, interp=interp)
        return augment(img)

    @staticmethod
    def random_zoom(img, max_size, factor=None, interp=BICUBIC):
        """Apply a random zoom augmentation to the given image."""
        augment = RandomZoomTransform(max_size, factor=factor, interp=interp)
        return augment(img)

    @staticmethod
    def crop(img, size, pos):
        """Crop the given image given a crop size and position."""
        augment = CropTransform(size=size, pos=pos)
        return augment(img)

    @staticmethod
    def patch(img, size, stride: int = 0):
        """Get a patch from the given image given a size and a stride."""
        augment = PatchTransform(size, stride=stride)
        return augment(img)

    @staticmethod
    def trim(img, size):
        """Trim the given image given a size."""
        augment = TrimTransform(size)
        return augment(img)

    @staticmethod
    def make_power_2(img, base: int = 4, interp=BICUBIC):
        """Make the image sides a multiple of a given base."""
        augment = MakePowerTwoTransform(base=base, interp=interp)
        return augment(img)

    @staticmethod
    def flip_img(img, flip):
        """Flip the given image."""
        augment = FlipTransform(flip=flip)
        return augment(img)

    @staticmethod
    def convert(img, grayscale: bool = False):
        """Convert image to a tensor."""
        augment = ConvertTransform(grayscale=grayscale)
        return augment(img)


class BaseTransform(ABC):
    """ Base class for all custom transforms"""
    def __init__(self):
        """Init"""

    @abstractmethod
    def __call__(self, sample):
        """Call"""

    def __repr__(self):
        return self.__class__.__name__ + "()"


class IdentityTransform(BaseTransform):
    """ Return Image unchanged"""
    def __call__(self, img):
        return img


class RandomZoomTransform(BaseTransform):
    """ Random Zoom Transform"""
    def __init__(self,
                 max_size: Tuple[int, int],
                 factor: bool = None,
                 interp=BICUBIC):
        super().__init__()
        self.max_size = max_size
        self.factor = factor
        self.interp = interp

    def __call__(self, img):
        """Call"""

        if self.factor is None:
            zoom_w, zoom_h = np.random.uniform(0.8, 1.0, size=[2])
        else:
            zoom_w, zoom_h = self.factor
        in_w, in_h = img.size
        max_w, max_h = self.max_size
        zoom_w = max(max_w, in_w * zoom_w)
        zoom_h = max(max_h, in_h * zoom_h)
        img = img.resize((int(round(zoom_w)), int(round(zoom_h))), self.interp)
        return img


class ConvertTransform(BaseTransform):
    """ Convert to Tensor and Normalize Transform"""
    def __init__(self, grayscale: bool = False):
        super().__init__()
        self.mean: Tuple[float, ...]
        self.std: Tuple[float, ...]
        if grayscale:
            self.mean, self.std = (0.5, ), (0.5, )
        else:
            self.mean, self.std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    def __call__(self, img):
        # Enforce single channel if img is grayscale
        if hasattr(img, "mode"):
            if img.mode == "L":
                self.mean, self.std = (0.5, ), (0.5, )
        elif hasattr(img, "shape"):
            if len(img.shape) == 2:
                self.mean, self.std = (0.5, ), (0.5, )

        # T.Normalize does not support PIL Image, convert to Torch Tensor first
        transform = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])

        return transform(img)


class ScaleWidthTransform(BaseTransform):
    """ Scale Width Transform"""
    def __init__(self, width: int, max_height: int, interp=BICUBIC):
        super().__init__()
        self.width = width
        self.max_height = max_height
        self.interp = interp

    def __call__(self, img):
        in_w, in_h = img.size
        if in_w == self.width and in_h >= self.max_height:
            return img
        width = self.width
        height = int(max(self.width * in_h / in_w, self.max_height))
        return img.resize((width, height), self.interp)


class ScaleShortsideTransform(BaseTransform):
    """ Scale Shortside Transform"""
    def __init__(self, size, interp=BICUBIC):
        super().__init__()
        self.size = size
        self.interp = interp

    def __call__(self, img):
        in_w, in_h = img.size
        shortside = min(in_w, in_h)
        if shortside >= self.size:
            return img
        scale = self.size / shortside
        width = round(in_w * scale)
        height = round(in_h * scale)
        return img.resize((width, height), self.interp)


class CropTransform(BaseTransform):
    """ Crop Transform"""
    def __init__(self,
                 size: Tuple[int, int] = None,
                 pos: Tuple[int, int] = None):
        super().__init__()
        self.size = size
        self.pos = pos

    def __call__(self, img):
        in_w, in_h = img.size
        pos_x, pos_y = self.pos
        out_w, out_h = self.size
        if in_w > out_w or in_h > out_h:
            return img.crop((pos_x, pos_y, pos_x + out_w, pos_y + out_h))
        return img


class PatchTransform(BaseTransform):
    """ Patch Transform"""
    def __init__(self, size: tuple, stride: int = 0):
        super().__init__()
        self.size = size
        self.stride = stride

    def __call__(self, img):
        in_w, in_h = img.size
        patch_w, patch_h = self.size
        num_w, num_h = in_w // patch_w, in_h // patch_h
        room_x = in_w - num_w * patch_w
        room_y = in_h - num_h * patch_h
        x_start = np.random.randint(int(room_x) + 1)
        y_start = np.random.randint(int(room_y) + 1)
        idx = self.stride % (num_w * num_h)
        idx_x, idx_y = idx // num_h, idx % num_w
        pos_x = x_start + idx_x * patch_w
        pos_y = y_start + idx_y * patch_h
        return img.crop((pos_x, pos_y, pos_x + patch_w, pos_y + patch_h))


class FlipTransform(BaseTransform):
    """ Flip Transform"""
    def __init__(self, flip: bool = True):
        super().__init__()
        self.flip = flip

    def __call__(self, img):
        if self.flip:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class TrimTransform(BaseTransform):
    """ Trim Transform"""
    def __init__(self, size: int = 256):
        super().__init__()
        self.size = size

    def __call__(self, img):
        in_w, in_h = img.size
        trim_w, trim_h = self.size
        if in_w > trim_w:
            start_x = np.random.randint(in_w - trim_w)
            end_x = start_x + trim_w
        else:
            start_x = 0
            end_x = in_w
        if in_h > trim_h:
            start_y = np.random.randint(in_h - trim_h)
            end_y = start_y + trim_h
        else:
            start_y = 0
            end_y = in_h
        return img.crop((start_x, start_y, end_x, end_y))


class MakePowerTwoTransform(BaseTransform):
    """ Make Power 2 Transform"""
    def __init__(self, base: int = 4, interp=BICUBIC):
        super().__init__()
        self.base = base
        self.interp = interp

    def __call__(self, img):
        in_w, in_h = img.size
        height = int(round(in_h / self.base) * self.base)
        width = int(round(in_w / self.base) * self.base)
        if height == in_h and width == in_w:
            return img
        print_size_warning(in_w, in_h, width, height)
        return img.resize((width, height), self.interp)


def print_size_warning(in_w, in_h, width, height):
    """Print warning information about image size (only print once)"""

    if not hasattr(print_size_warning, "has_printed"):
        print(
            f"The image size needs to be a multiple of 4. "
            f"The loaded image size was ({in_w}, {in_h}), so it was adjusted "
            f"to ({width}, {height}). This adjustment will be done to all "
            f"images whose sizes are not multiples of 4")
        print_size_warning.has_printed = True
