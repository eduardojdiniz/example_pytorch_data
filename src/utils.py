#!/usr/bin/env python
# coding=utf-8

import hashlib
import os.path
import traceback
from os.path import join as pjoin
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import torch
import torchvision.transforms as T  # type: ignore
from PIL import Image

plt.rcParams["savefig.bbox"] = "tight"

_img_t = Union[Image.Image, torch.Tensor]  # pylint: disable=C0103

_imgs_t = List[Union[_img_t, List[_img_t]]]  # pylint: disable=C0103

__all__ = [
    "IMG_EXTENSIONS",
    "DATA_DIR",
    "SAVE_DIR",
    "is_valid_extension",
    "default_image_loader",
    "denormalize",
    "plot",
    "load_standard_test_imgs",
    "check_integrity",
]

IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".pgm",
)

DATA_DIR = Path.cwd().parent / "data"
SAVE_DIR = Path.cwd().parent / "trials"


def is_valid_extension(filename: Path, extensions: Tuple[str, ...]) -> bool:
    """
    Verifies if given file name has a valid extension.

    Parameters
    ----------
    filename : str
        Path to a file
    extensions : Tuple[str, ...]
        Extensions to consider (lowercase)

    Returns
    -------
    return : bool
        True if the filename ends with one of given extensions
    """

    fname = str(filename)
    return any(fname.lower().endswith(ext) for ext in extensions)


def is_valid_image(filename: Path) -> bool:
    """
    Verifies if given file name has a valid image extension

    Parameters
    ----------
    filename : str
        Path to a file

    Returns
    -------
    return : bool
        True if the filename ends with one of the valid image extensions
    """
    return is_valid_extension(filename, IMG_EXTENSIONS)


def default_image_loader(path: Path) -> Image.Image:
    """
    Load image file as RGB PIL Image

    Parameters
    ----------
    path : Path
        Image file path

    Returns
    -------
    return : Image.Image
       RGB PIL Image
    """

    # Open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with path.open(mode="rb") as fid:
        img = Image.open(fid)
        return img.convert("RGB")


def denormalize(tensor: torch.Tensor,
                mean: Tuple[float, ...] = None,
                std: Tuple[float, ...] = None):
    """
    Undoes mean/standard deviation normalization, zero to one scaling, and
    channel rearrangement for a batch of images.

    Parameters
    ----------
    tensor : torch.Tensor
        A (CHANNELS x HEIGHT x WIDTH) tensor
    mean: Tuple[float, ...]
        A tuple of mean values to be subtracted from each image channel.
    std: Tuple[float, ...]
        A tuple of standard deviation values to be devided from each image
        channel.

    Returns
    ----------
    array : numpy.ndarray[float]
        A (HEIGHT x WIDTH x CHANNELS) numpy array of floats
    """
    if not mean:
        if tensor.shape[0] == 1:
            mean = (-0.5 / 0.5, )
        else:
            mean = (-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5)
    if not std:
        if tensor.shape[0] == 1:
            std = (1 / 0.5, )
        else:
            std = (1 / 0.5, 1 / 0.5, 1 / 0.5)
    inverse_normalize = T.Normalize(mean=mean, std=std)
    denormalized_tensor = (inverse_normalize(tensor) * 255.0).type(torch.uint8)
    array = denormalized_tensor.permute(1, 2, 0).numpy().squeeze()
    return array


def plot(imgs: _imgs_t,
         baseline_imgs: Union[_img_t, _imgs_t] = None,
         row_title: List[str] = None,
         title: str = None,
         **imshow_kwargs) -> None:
    """
    Plot images in a 2D grid.

    Arguments
    ---------
    imgs : _imgs_t
        List of images to be plotted. `imgs` is either a list of
        `_img_t` images or a list of lists of `_img_t` images. Either way,
        each element of `imgs` holds a row of the image grid to be plotted.
    baseline_imgs : Union[_img_t, _imgs_]
        List of baseline images. If not None, the first column of the grid will
        be filled with the baseline images.`baseline_imgs` is either
        a single `_img_t` image, or a list of `_img_t` images of the same
        length of an element of `imgs`.
    row_title : List[str]
        List of row titles. If not None, `len(row_title)` must be equal to
        `len(imgs)`.

    Types
    -----
    _img_t = Union[PIL.Image.Image, torch.Tensor]
    _imgs_t = List[Union[_img_t, List[_img_t]]]
    """

    # Make a 2d grid even if there's just 1 row
    if not isinstance(imgs[0], list):
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])

    if not baseline_imgs:
        with_baseline = False
    else:
        if not isinstance(baseline_imgs, list):
            baseline_imgs = [baseline_imgs for i in range(0, num_rows)]
        else:
            if len(baseline_imgs) == 1:
                baseline_imgs = [baseline_imgs[0] for i in range(0, num_rows)]
            elif len(baseline_imgs) != num_rows:
                msg = (
                    "Number of elements in `baseline_imgs` ",
                    "must match the number of elements in `imgs[0]`",
                )
                raise ValueError(msg)
            if isinstance(baseline_imgs[0], list):
                msg = (
                    "Elements of `baseline_imgs` must be PIL Images ",
                    "or Torch Tensors",
                )
                raise TypeError(msg)
        with_baseline = True
        num_cols += 1  # First column is now the baseline images
    if row_title:
        if len(row_title) != num_rows:
            msg = (
                "Number of elements in `row_title` ",
                "must match the number of elements in `imgs`",
            )
            raise ValueError(msg)

    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [baseline_imgs[row_idx]] + row if with_baseline else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            if isinstance(img, torch.Tensor):
                img = denormalize(img)
            else:
                img = np.asarray(img)
            if len(img.shape) == 2:
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            else:
                ax.imshow(img, **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_baseline:
        plt.sca(axs[0, 0])
        plt.title(label="Baseline images", size=15)

    if row_title is not None:
        for row_idx in range(num_rows):
            plt.sca(axs[row_idx, 0])
            plt.ylabel(row_title[row_idx], rotation=0, labelpad=50, size=15)
            plt.tight_layout()

    if title:
        fig.suptitle(t=title, size=16)

    fig.tight_layout()
    return fig


def load_standard_test_imgs(directory: Path = Path().cwd() / "imgs"):
    directory = directory.expanduser()
    test_imgs = []
    names = []
    for root, _, fnames in sorted(os.walk(directory, followlinks=True)):
        for fname in sorted(fnames):
            if is_valid_image(Path(fname)):
                path = pjoin(root, fname)
                test_imgs.extend([Image.open(path)])
                names.append(Path(path).stem)
    return test_imgs, names


def calculate_md5_dir(dirpath: Path,
                      chunk_size: int = 1024 * 1024,
                      verbose: bool = False) -> str:
    md5 = hashlib.md5()
    try:
        for root, _, files in sorted(os.walk(dirpath)):
            for name in files:
                if verbose:
                    print("Hashing", name)
                fpath = Path(root) / name
                with fpath.open(mode="rb") as fid:
                    for chunk in iter(lambda: fid.read(chunk_size), b""):
                        md5.update(chunk)

    except BaseException:
        # Print the stack traceback
        traceback.print_exc()
        return -2

    return md5.hexdigest()


def calculate_md5_file(fpath: Path, chunk_size: int = 1024 * 1024) -> str:
    md5 = hashlib.md5()
    try:
        with fpath.open(mode="rb") as fid:
            for chunk in iter(lambda: fid.read(chunk_size), b""):
                md5.update(chunk)
    except BaseException:
        # Print the stack traceback
        traceback.print_exc()
        return -2
    return md5.hexdigest()


def check_md5(path: Path, md5: str, **kwargs: Any) -> bool:
    if path.is_dir():
        return md5 == calculate_md5_dir(path, **kwargs)
    return md5 == calculate_md5_file(path, **kwargs)


def check_integrity(path: Path, md5: Optional[str] = None) -> bool:
    if not os.path.exists(path):
        return False
    if md5 is None:
        return True
    return check_md5(path, md5)
