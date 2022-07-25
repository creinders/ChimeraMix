from abc import abstractmethod

import torch
from pytorch_lightning.core import LightningDataModule
from torch.utils.data.dataset import Dataset

from datasets.pipeline.IndexDataset import IndexDataset
from datasets.pipeline.MaximumLabelDataset import MaximumLabelDataset
from datasets.pipeline.MultiTransformerDataset import MultiTransformerDataset
from datasets.pipeline.PairDataset import PairDataset
from datasets.pipeline.RepeatDataset import RepeatDataset
from datasets.pipeline.SegmentationDataset import SegmentationDataset
from datasets.pipeline.TargetDataset import TargetDataset
from utils.transforms import get_test_transforms, get_train_transforms


class VisionDataModule(LightningDataModule):
    pass

    def __init__(
        self,
        name,
        batch_size=64,
        max_labels_per_class=None,
        max_labels_per_class_seed=None,
        num_train_dataset_repeats=None,
        pairs_train=False,
        pairs_train_kwd=None,
        pairs_test=False,
        pairs_test_kwd=None,
        transforms_train=None,
        train_shuffle=True,
        transforms_test=None,
        index_train=False,
        validation_enabled=True,
        validation_add_train_dataset=False,
        train_workers=8,
        val_workers=8,
        segmentation=False,
        segmentation_method=None,
        segmentation_kwargs=None,
    ):
        super().__init__()
        self.name = name
        self.batch_size = batch_size

        if segmentation_method is None:
            segmentation_method = "felzenszwalb"
        if segmentation_kwargs is None:
            segmentation_kwargs = {"scale": 60, "min_size": 60}

        if transforms_train is None:
            transforms_train = get_train_transforms(dataset=name)

        if transforms_test is None:
            transforms_test = get_test_transforms(dataset=name)

        self.max_labels_per_class = max_labels_per_class
        self.max_labels_per_class_seed = max_labels_per_class_seed
        self.num_train_dataset_repeats = num_train_dataset_repeats
        self.pairs_train = pairs_train
        self.pairs_train_kwd = pairs_train_kwd
        self.pairs_test = pairs_test
        self.pairs_test_kwd = pairs_test_kwd
        self.index_train = index_train
        self.transforms_train = transforms_train
        self.train_shuffle = train_shuffle
        self.transforms_test = transforms_test
        self.validation_enabled = validation_enabled
        self.validation_add_train_dataset = validation_add_train_dataset
        self.train_workers = train_workers
        self.val_workers = val_workers
        self.segmentation = segmentation
        self.segmentation_method = segmentation_method
        self.segmentation_kwargs = segmentation_kwargs

        self.train_ds = None
        self.val_ds = None
        self.train_loader = None
        self.val_loader = None

    @abstractmethod
    def get_train_dataset(self, transform) -> Dataset:
        pass

    @abstractmethod
    def get_test_dataset(self, transform) -> Dataset:
        pass

    def generate_dataloader(
        self,
        transforms,
        get_dataset,
        shuffle,
        max_labels_per_class=None,
        index=None,
        num_dataset_repeats=None,
        pairs=None,
        pairs_kwd=None,
        workers=8,
    ):

        dataset = get_dataset(transform=None)

        dataset = TargetDataset(dataset)

        dataset = MultiTransformerDataset(dataset, transformers=transforms)

        if max_labels_per_class is not None and max_labels_per_class > 0:
            dataset = MaximumLabelDataset(
                dataset, max_labels_per_class, seed=self.max_labels_per_class_seed
            )

        if self.segmentation:
            dataset = SegmentationDataset(
                dataset,
                method=self.segmentation_method,
                method_kwargs=self.segmentation_kwargs,
            )

        if index:
            dataset = IndexDataset(dataset)

        if pairs:
            pairs_kwd = pairs_kwd or {}
            dataset = PairDataset(dataset, **pairs_kwd)

        if num_dataset_repeats is not None and num_dataset_repeats > 1:
            dataset = RepeatDataset(
                dataset,
                num_dataset_repeats,
                seed=self.max_labels_per_class_seed + 1000
                if self.max_labels_per_class_seed is not None
                else None,
            )

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=workers,
            sampler=None,
            pin_memory=True,
        )

        return dataset, dataloader

    def _generate_train_dataloader(self, shuffle=True, num_train_dataset_repeats=None):
        return self.generate_dataloader(
            transforms=self.transforms_train,
            get_dataset=self.get_train_dataset,
            shuffle=shuffle,
            max_labels_per_class=self.max_labels_per_class,
            index=self.index_train,
            num_dataset_repeats=num_train_dataset_repeats
            if num_train_dataset_repeats is not None
            else self.num_train_dataset_repeats,
            pairs=self.pairs_train,
            pairs_kwd=self.pairs_train_kwd,
            workers=self.train_workers,
        )

    def setup(self, stage):

        train_ds, train_loader = self._generate_train_dataloader(
            shuffle=self.train_shuffle
        )

        val_ds, val_loader = self.generate_dataloader(
            transforms=self.transforms_test,
            get_dataset=self.get_test_dataset,
            shuffle=False,
            pairs=self.pairs_test,
            pairs_kwd=self.pairs_test_kwd,
            workers=self.val_workers,
        )
        self.train_ds = train_ds
        self.val_ds = val_ds

        # assign to use in dataloaders
        self.train_loader = train_loader
        self.val_loader = val_loader if self.validation_enabled else None
        # self.test_dataset = mnist_test

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):

        if self.validation_add_train_dataset:
            train_loader = self._generate_train_dataloader(
                shuffle=False, num_train_dataset_repeats=1
            )

            return [self.val_loader, train_loader]
        else:
            return self.val_loader

