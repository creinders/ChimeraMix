import torchvision.datasets as datasets
from datasets.vision_data_module import VisionDataModule
from datasets.cifair import ciFAIR100


class CIFAR100DataModule(VisionDataModule):
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
        use_cifair=True,
        *args,
        **kwargs
    ):

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

        self.num_classes = 100
        self.batch_size = batch_size
        self.data_path = data_path

        self.use_cifair = use_cifair
        self.train_ds = None
        self.test_ds = None

    def get_classes(self):
        ds = self.train_ds if self.train_ds else self.test_ds
        return ds.classes

    def prepare_data(self):
        # download only
        datasets.CIFAR100(self.data_path, train=True, download=True)
        datasets.CIFAR100(self.data_path, train=False, download=True)

    def get_train_dataset(self, transform):
        self.train_ds = datasets.CIFAR100(
            root=self.data_path, train=True, download=True, transform=transform
        )
        return self.train_ds

    def get_test_dataset(self, transform):
        if self.use_cifair:
            self.test_ds = ciFAIR100(
                root=self.data_path, train=False, download=True, transform=transform
            )
        else:
            self.test_ds = datasets.CIFAR100(
                root=self.data_path, train=False, download=True, transform=transform
            )
        return self.test_ds
