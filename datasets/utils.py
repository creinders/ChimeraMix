from functools import partial


def get_dm_class(cfg):

    if cfg.dataset == "cifar10" or cfg.dataset == "cifair10":
        from datasets.cifar10 import CIFAR10DataModule

        return partial(CIFAR10DataModule, use_cifair=cfg.dataset == "cifair10")
    elif cfg.dataset == "cifar100" or cfg.dataset == "cifair100":
        from datasets.cifar100 import CIFAR100DataModule

        return partial(CIFAR100DataModule, use_cifair=cfg.dataset == "cifair100")
    elif cfg.dataset == "stl10":
        from datasets.stl10 import STL10DataModule

        return STL10DataModule

    else:
        raise ValueError("unknown dataset", cfg.dataset)
