import copy

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

from utils.cutout import Cutout
from utils.random_erase import RandomErasing


class ComposeDict:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, item):
        image = item["image"]

        for t in self.transforms:
            image = t(image)

        out_dict = copy.copy(item)
        out_dict["image"] = image

        return out_dict


def get_train_transforms(
    dataset,
    image_size=None,
    mean=None,
    std=None,
    aug_flipping=True,
    aug_cutout=None,
    aug_random_erase=None,
    aug_autoaugment=None,
    aug_autoaugment_policy=None,
    aug_trivialaugment=None,
    aug_trivialaugment_kwargs=None,
    process_mode="dict",
):
    assert dataset in ["cifar10", "cifair10", "cifar100", "cifair100", "stl10"]

    if image_size is None:
        image_size = get_default_image_size(dataset)
    if mean is None or std is None:
        mean, std = get_normalization_params(dataset)

    transform_list = []

    if aug_autoaugment:
        if aug_autoaugment_policy is None:
            aug_autoaugment_policy = get_default_autoaugment_policy(dataset)
        transform_list += [
            torchvision.transforms.AutoAugment(policy=aug_autoaugment_policy)
        ]

    if aug_trivialaugment:
        if aug_trivialaugment_kwargs is None:
            aug_trivialaugment_kwargs = {
                "num_magnitude_bins": 31,
                "interpolation": InterpolationMode.NEAREST,
                "fill": None,
            }
        transform_list += [
            torchvision.transforms.TrivialAugmentWide(**aug_trivialaugment_kwargs)
        ]

    if dataset in ["cifar10", "cifair10", "cifar100", "cifair100"]:
        transform_list += [
            transforms.Resize(image_size),
            transforms.RandomCrop(
                image_size, padding=image_size // 8, padding_mode="reflect"
            ),
        ]
    elif dataset == "stl10":
        transform_list += [
            transforms.RandomCrop(96, padding=12, padding_mode="reflect"),
        ]
    else:
        raise ValueError

    if aug_flipping:
        transform_list += [
            transforms.RandomHorizontalFlip(),
        ]

    transform_list += [
        transforms.ToTensor(),
    ]

    if aug_random_erase:
        transform_list += [RandomErasing()]

    transform_list += [
        transforms.Normalize(mean, std),
    ]

    if aug_cutout:
        transform_list += [Cutout(dataset=dataset)]

    metadata = {"image_size": image_size, "mean": mean, "std": std}

    if process_mode == "dict":
        return ComposeDict(transform_list), metadata
    else:
        return transforms.Compose(transform_list), metadata


def get_test_transforms(
    dataset, image_size=None, mean=None, std=None, process_mode="dict"
):
    assert dataset in ["cifar10", "cifair10", "cifar100", "cifair100", "stl10"]

    if image_size is None:
        image_size = get_default_image_size(dataset)
    if mean is None or std is None:
        mean, std = get_normalization_params(dataset)

    if dataset in ["cifar10", "cifair10", "cifar100", "cifair100"]:
        transform_list = [
            transforms.Resize(image_size),
        ]
    elif dataset == "stl10":
        transform_list = []
    else:
        raise ValueError

    transform_list += [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]

    if process_mode == "dict":
        return ComposeDict(transform_list)
    else:
        return transforms.Compose(transform_list)


def get_default_image_size(dataset):

    if dataset in ["cifar10", "cifair10", "cifar100", "cifair100"]:
        return 32
    elif dataset == "stl10":
        return 96
    else:
        raise ValueError


def get_default_autoaugment_policy(dataset):
    from torchvision.transforms import AutoAugmentPolicy

    if dataset in ["cifar10", "cifair10", "cifar100", "cifair100"]:
        return AutoAugmentPolicy.CIFAR10
    elif dataset == "stl10":
        return AutoAugmentPolicy.IMAGENET
    else:
        raise ValueError


def cifar10_normalization_params():
    mean = [x / 255.0 for x in [125.3, 123.0, 113.9]]
    std = [x / 255.0 for x in [63.0, 62.1, 66.7]]
    return mean, std


def stl10_normalization_params():
    return ((0.43, 0.42, 0.39), (0.27, 0.26, 0.27))


def get_normalization_params(dataset):
    if dataset in ["cifar10", "cifair10", "cifar100", "cifair100"]:
        return cifar10_normalization_params()
    elif dataset == "stl10":
        return stl10_normalization_params()
    else:
        raise ValueError
