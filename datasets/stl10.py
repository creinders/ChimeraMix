import torchvision.datasets as datasets

from .vision_data_module import VisionDataModule


class STL10DataModule(VisionDataModule):
    def __init__(
        self,
        batch_size=64,
        max_labels_per_class=None,
        max_labels_per_class_seed=None,
        num_train_dataset_repeats=None,
        pairs_train=False,
        pairs_train_kwd=None,
        pairs_test=False,
        pairs_test_kwd=None,
        transforms_train=None,
        transforms_test=None,
        validation_enabled=True,
        index_train=False,
        validation_add_train_dataset=False,
        train_workers=8,
        val_workers=8,
        segmentation=False,
        segmentation_method=None,
        segmentation_kwargs={"scale": 400, "min_size": 400},
        data_path="tmp/data",
        *args,
        **kwargs
    ):
        super().__init__(
            name="stl10",
            batch_size=batch_size,
            max_labels_per_class=max_labels_per_class,
            max_labels_per_class_seed=max_labels_per_class_seed,
            num_train_dataset_repeats=num_train_dataset_repeats,
            pairs_train=pairs_train,
            pairs_train_kwd=pairs_train_kwd,
            pairs_test=pairs_test,
            pairs_test_kwd=pairs_test_kwd,
            transforms_train=transforms_train,
            transforms_test=transforms_test,
            validation_enabled=validation_enabled,
            index_train=index_train,
            validation_add_train_dataset=validation_add_train_dataset,
            train_workers=train_workers,
            val_workers=val_workers,
            segmentation=segmentation,
            segmentation_method=segmentation_method,
            segmentation_kwargs=segmentation_kwargs,
            *args,
            **kwargs
        )
        self.num_classes = 10
        self.batch_size = batch_size
        self.data_path = data_path

        self.train_ds = None
        self.test_ds = None
        self.classes = None

    def get_classes(self):
        ds = self.train_ds if self.train_ds else self.test_ds
        return ds.classes

    def prepare_data(self):
        # download only
        datasets.STL10(self.data_path, split="train", download=True)
        datasets.STL10(self.data_path, split="test", download=True)

    def get_train_dataset(self, transform):
        self.train_ds = datasets.STL10(
            root=self.data_path, split="train", download=False, transform=transform
        )
        self.classes = self.train_ds.classes
        return self.train_ds

    def get_test_dataset(self, transform):
        self.test_ds = datasets.STL10(
            root=self.data_path, split="test", download=False, transform=transform
        )
        return self.test_ds

