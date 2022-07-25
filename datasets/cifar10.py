import torchvision.datasets as datasets
from torch.utils.data import Dataset

from datasets.cifair import ciFAIR10
from datasets.vision_data_module import VisionDataModule


class CIFAR10DataModule(VisionDataModule):
    def __init__(
        self,
        batch_size=64,
        image_size=None,
        max_labels_per_class=None,
        max_labels_per_class_seed=None,
        num_train_dataset_repeats=None,
        pairs_train=False,
        pairs_train_kwd=None,
        pairs_test=False,
        pairs_test_kwd=None,
        index_train=False,
        transforms_train=None,
        transforms_test=None,
        validation_enabled=True,
        validation_add_train_dataset=False,
        train_workers=8,
        val_workers=8,
        segmentation=False,
        segmentation_method="felzenszwalb",
        segmentation_kwargs={"scale": 60, "min_size": 60},
        data_path="tmp/data",
        use_cifair=False,
        *args,
        **kwargs
    ):

        if not image_size:
            image_size = 32
        name = "cifair10" if use_cifair else "cifar10"
        super().__init__(
            name=name,
            batch_size=batch_size,
            max_labels_per_class=max_labels_per_class,
            max_labels_per_class_seed=max_labels_per_class_seed,
            num_train_dataset_repeats=num_train_dataset_repeats,
            pairs_train=pairs_train,
            pairs_train_kwd=pairs_train_kwd,
            pairs_test=pairs_test,
            pairs_test_kwd=pairs_test_kwd,
            index_train=index_train,
            transforms_train=transforms_train,
            transforms_test=transforms_test,
            validation_enabled=validation_enabled,
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

        self.image_size = image_size

        self.max_labels_per_class = max_labels_per_class
        self.num_train_dataset_repeats = num_train_dataset_repeats
        self.pairs_train = pairs_train
        self.pairs_test = pairs_test
        self.use_cifair = use_cifair
        self.classes = None

    def prepare_data(self):
        # download only
        datasets.CIFAR10(self.data_path, train=True, download=True)
        datasets.CIFAR10(self.data_path, train=False, download=True)

    def get_train_dataset(self, transform) -> Dataset:
        dataset = datasets.CIFAR10(
            root=self.data_path, train=True, download=True, transform=transform
        )
        self.classes = dataset.classes
        return dataset

    def get_test_dataset(self, transform) -> Dataset:
        if self.use_cifair:
            return ciFAIR10(
                root=self.data_path, train=False, download=True, transform=transform
            )
        else:
            return datasets.CIFAR10(
                root=self.data_path, train=False, download=True, transform=transform
            )
