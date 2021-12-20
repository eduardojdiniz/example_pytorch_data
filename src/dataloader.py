#!/usr/bin/env python
# coding=utf-8
"""
Dataloaders are based on the PyTorch ``torch.utils.data.Dataloader`` data
primitive. They are wrappers around ``torch.utils.data.Dataset`` that enable
easy access to the dataset samples, i.e., they prepare your data for
training/testing. Specifically, dataloaders are iterables that abstracts the
complexity of retrieving "minibatches" from Datasets, reshuffling the data at
every epoch to reduce model overfitting, use Python's ``multiprocessing``
to speed up data retrieval, and automatic memory pinning, in an easy API.
"""

from typing import Any, Callable, List, TypeVar, Iterator, Tuple

import numpy as np
from sklearn.model_selection import KFold  # type: ignore
from torch.utils.data import DataLoader
from torch.utils.data import Sampler, SubsetRandomSampler, random_split
from dataset import BaseVisionDataset

Type = TypeVar("Type")
WorkerInitFnType = Callable[[int], None]

# Ideally we would parameterize `DataLoader` by the return type of
# `collate_fn`, but there is currently no way to have that type parameter set
# to a default value if the user doesn't pass in a custom 'collate_fn'.
# See https://github.com/python/mypy/issues/3737.
CollateFnType = Callable[[List[Type]], Any]

__all__ = ["BaseDataLoader", "KFoldDataLoader", "OneFoldDataLoader"]


class BaseDataLoader(DataLoader):
    """Base class for all data loaders"""
    def __init__(
        self,
        dataset: BaseVisionDataset,
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: Sampler = None,
        num_workers: int = 0,
        collate_fn: CollateFnType = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        worker_init_fn: WorkerInitFnType = None,
        generator=None,
        train: bool = True,
    ) -> None:

        drop_last = bool(train)

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            worker_init_fn=worker_init_fn,
            generator=generator,
        )


class KFoldDataLoader:
    """Create train and validation dataloaders for K-Fold Cross-Validation"""
    def __init__(
        self,
        dataset: BaseVisionDataset,
        batch_size: int = 1,
        num_workers: int = 0,
        collate_fn: CollateFnType = None,
        pin_memory: bool = False,
        worker_init_fn: WorkerInitFnType = None,
        generator=None,
        n_splits: int = 5,
        kfold_shuffle: bool = True,
        random_state: int = 0,
    ) -> None:

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.worker_init_fn = worker_init_fn
        self.generator = generator
        self.drop_last = True
        self.kfold = KFold(n_splits=n_splits,
                           shuffle=kfold_shuffle,
                           random_state=random_state)
        self.fold: int
        self.kfold_split: Iterator[np.ndarray]

    def __iter__(self):
        self.fold = 0
        self.kfold_split = self.kfold.split(self.dataset)
        return self

    def __next__(self) -> Tuple[int, BaseDataLoader, BaseDataLoader]:
        train_idx, val_idx = next(self.kfold_split)
        self.fold += 1
        train_subsampler = SubsetRandomSampler(train_idx)
        val_subsampler = SubsetRandomSampler(val_idx)
        return (
            self.fold,
            self.get_dataloader(train_subsampler),
            self.get_dataloader(val_subsampler),
        )

    def get_dataloader(self, sampler: Sampler) -> BaseDataLoader:
        """
        Get train or validation dataloaders.

        Parameters
        ----------
        sampler : Sampler
            Sampler for either train or validation samples.

        Returns
        -------
        _ : BaseDataloader
            Train or validation dataloader.
        """
        return BaseDataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            worker_init_fn=self.worker_init_fn,
            generator=self.generator,
        )


class OneFoldDataLoader:
    """Random split dataset into train and validation dataloaders"""
    def __init__(
        self,
        dataset: BaseVisionDataset,
        batch_size: int = 1,
        num_workers: int = 0,
        collate_fn: CollateFnType = None,
        pin_memory: bool = False,
        worker_init_fn: WorkerInitFnType = None,
        generator=None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.worker_init_fn = worker_init_fn
        self.generator = generator
        self.drop_last = True

    def __call__(self, val_split: float = 0.2):
        split = int(np.floor(val_split * len(self.dataset)))
        sizes = [len(self.dataset) - split, split]
        subset_list: list = random_split(self.dataset,
                                         sizes,
                                         generator=self.generator)

        return (self.get_dataloader(subset) for subset in subset_list)

    def get_dataloader(self, subset: BaseVisionDataset):
        """
        Get train or validation dataloaders.

        Parameters
        ----------
        subset : BaseVisionDataset
            Subset of samples.

        Returns
        -------
        _ : BaseDataloader
            Train or validation dataloader.
        """
        return BaseDataLoader(
            dataset=subset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            worker_init_fn=self.worker_init_fn,
            generator=self.generator,
        )
