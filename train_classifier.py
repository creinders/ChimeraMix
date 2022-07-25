import atexit
import math
import random

import hydra
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.utils.data.distributed
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.core import LightningModule
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.confusion_matrix import ConfusionMatrix
from torchvision.models import resnet50
from torchvision.utils import make_grid

from datasets.utils import get_dm_class
from models.wide_resnet import WideResNet
from utils.checkpoints import checkpoint_chimera, checkpoint_cls
from utils.eta_callback import ETACallback
from utils.normalize_inverse import NormalizeInverse
from utils.tensor_utils import linear_interpolation, resize_to
from utils.transforms import get_test_transforms, get_train_transforms


class ClassifierLightningModel(LightningModule):
    def __init__(self, image_size, num_classes, mean, std, module_mix, **hparams):
        super().__init__()

        self.params = DictConfig(hparams)

        mix_modes = {
            "image_a": 1.0,
            "mix_generator": 0.0,
            "mixup": 0.0,
        }

        if self.params.mixup:
            mix_modes["image_a"] = 0.0
            mix_modes["mixup"] = 1.0

        if module_mix:
            self.params.mix_size = module_mix.params.mix_size
            self.params.noise_interpolation_mode = (
                module_mix.params.noise_interpolation_mode
            )
            mix_modes["mix_generator"] = 1.0

        self.params.mix_modes = mix_modes
        self.save_hyperparameters()

        self.image_size = image_size
        self.num_classes = num_classes
        self.mean = mean
        self.std = std

        self.module_mix = module_mix

        self.mix_mode_keys = list(mix_modes.keys())
        self.mix_mode_probs = np.array(list(mix_modes.values()))
        self.mix_mode_probs = self.mix_mode_probs / np.sum(self.mix_mode_probs)
        print(self.mix_mode_keys)
        print(self.mix_mode_probs)

        if self.params.model == "wideresnet":
            model = WideResNet(
                depth=28, num_classes=self.num_classes, widen_factor=10, dropRate=0.3
            )
        elif self.params.model == "wideresnet_16_8":
            model = WideResNet(
                depth=16, num_classes=self.num_classes, widen_factor=8, dropRate=0.3
            )
        elif self.params.model == "resnet50":
            model = resnet50(pretrained=self.params.pretrained)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, self.num_classes)
        else:
            raise ValueError("unknown model", self.params.model)

        self.model = model

        self.criterion = nn.CrossEntropyLoss()

        self.train_accuracy = Accuracy()
        self.train_accuracy_macro = Accuracy(average="macro", num_classes=num_classes)
        self.train_accuracy_top5 = Accuracy(top_k=5)
        self.val_accuracy = Accuracy()
        self.val_accuracy_macro = Accuracy(average="macro", num_classes=num_classes)
        self.val_accuracy_top5 = Accuracy(top_k=5)
        self.val_confusion = ConfusionMatrix(num_classes=num_classes)

        self.normalize_inv = NormalizeInverse(self.mean, self.std)

    def forward(self, x):
        return self.model(x)

    def on_epoch_start(self) -> None:

        if self.current_epoch % 1 == 0:
            self.log_next_batch = True
            self.log_next_cam = True
        self.val_confusion.reset()

    def training_step(self, batch, batch_idx):

        dict_a, dict_b = batch
        idx_a, images_a, target_a = dict_a["idx"], dict_a["image"], dict_a["target"]
        idx_b, images_b, target_b = dict_b["idx"], dict_b["image"], dict_b["target"]
        if self.params.mix_mode == "segmentation":
            segmentation_a = dict_a["segmentation"]
            segmentation_b = dict_b["segmentation"]

        target_a = target_a.squeeze()
        bs = images_a.size(0)

        mix_mode = np.random.choice(self.mix_mode_keys, p=self.mix_mode_probs)

        if mix_mode == "mixup":
            alpha = 1.0
            f = np.random.beta(alpha, alpha)
            f = torch.tensor([f]).to(self.device)

            index = torch.randperm(bs).to(self.device)

            f = f.view(1, 1, 1, 1)
            images_mix = linear_interpolation(images_a, images_a[index], f=f)
            images_b = images_a[index]
            target_b = target_a[index]
        elif mix_mode == "mix_generator":
            self.module_mix.eval()
            if self.params.mix_discriminator_filter_repeats > 1:
                target_a_oh = F.one_hot(target_a, num_classes=self.num_classes).float()
                if self.params.mix_mode == "segmentation":
                    noise = self.module_mix.gen_noise(
                        bs * self.params.mix_discriminator_filter_repeats,
                        segmentation_a.tile(
                            self.params.mix_discriminator_filter_repeats, 1, 1
                        ),
                        segmentation_b.tile(
                            self.params.mix_discriminator_filter_repeats, 1, 1
                        ),
                    )
                else:
                    noise = self.module_mix.gen_noise(
                        bs * self.params.mix_discriminator_filter_repeats
                    )
                images_mix, _, _, _ = self.module_mix(
                    images_a.tile(
                        self.params.mix_discriminator_filter_repeats, 1, 1, 1
                    ),
                    images_b.tile(
                        self.params.mix_discriminator_filter_repeats, 1, 1, 1
                    ),
                    noise,
                )
                disc_probs = (
                    self.module_mix.netD(images_mix, target_a_oh)
                    .squeeze()
                    .mean(dim=[-1, -2])
                )
                disc_probs_argmin = torch.argmin(
                    disc_probs.view(self.params.mix_discriminator_filter_repeats, bs),
                    dim=0,
                )
                images_mix = images_mix.view(
                    self.params.mix_discriminator_filter_repeats,
                    bs,
                    *images_mix.shape[1:],
                )
                images_mix = images_mix[disc_probs_argmin, torch.arange(bs)]
            else:
                images_a_resized = resize_to(
                    images_a, target_size=self.module_mix.image_size
                )
                images_b_resized = resize_to(
                    images_b, target_size=self.module_mix.image_size
                )
                
                f_a = self.module_mix.generator.encode(images_a_resized)
                f_b = self.module_mix.generator.encode(images_b_resized)

                if self.params.mix_mode == "segmentation":
                    segmentation_a_resized = resize_to(
                        segmentation_a,
                        target_size=self.module_mix.image_size,
                        mode="nearest",
                        add_channel=True,
                    )
                    segmentation_b_resized = resize_to(
                        segmentation_b,
                        target_size=self.module_mix.image_size,
                        mode="nearest",
                        add_channel=True,
                    )

                    mask = self.module_mix.gen_noise(
                        bs,
                        segmentation_a=segmentation_a_resized,
                        segmentation_b=segmentation_b_resized,
                        resize=not self.params.mix_direct,
                    )
                else:
                    mask = self.module_mix.gen_noise(
                        bs, resize=not self.params.mix_direct
                    )

                if self.params.mix_direct:
                    mask_a = mask[:, 0, ...][:, None]
                    mask_b = mask[:, 1, ...][:, None]

                    images_mix = mask_a * images_a_resized + mask_b * images_b_resized
                else:
                    images_mix, _, _, _ = self.module_mix.generator.mixDecode(
                        f_a, f_b, mask
                    )

                images_mix = resize_to(images_mix, target_size=self.image_size)

        elif mix_mode == "image_a":
            images_mix = images_a
        else:
            raise ValueError("unknown mix mode", mix_mode)

        target = target_a

        output = self.model(images_mix)

        if mix_mode == "mixup":
            loss = (1 - f) * self.criterion(output, target_a) + f * self.criterion(
                output, target_b
            )
        else:
            loss = self.criterion(output, target)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        if mix_mode not in ["mixup"]:
            self.train_accuracy(F.softmax(output, dim=1), target)
            self.train_accuracy_macro(F.softmax(output, dim=1), target)
            self.train_accuracy_top5(F.softmax(output, dim=1), target)

            self.log("train_acc", self.train_accuracy, on_step=True, on_epoch=True)
            self.log(
                "train_acc_macro",
                self.train_accuracy_macro,
                on_step=True,
                on_epoch=True,
            )
            self.log(
                "train_acc_top5", self.train_accuracy_top5, on_step=False, on_epoch=True
            )

        if self.log_next_batch:
            n = 8
            images_a = images_a[:n]
            images_mix = images_mix[:n]

            image_list = [images_a]
            if images_b is not None:
                images_b = images_b[:n]
                image_list.append(images_b)

            image_list.append(images_mix)

            data = torch.cat(image_list, dim=0)
            data = self.normalize_inv(data)

            self.logger.experiment.log(
                {
                    "batch": wandb.Image(
                        make_grid(data.float(), normalize=False, nrow=8,), mode="RGB",
                    ),
                },
                commit=False,
            )
            self.log_next_batch = False

        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        target = batch["target"]

        output = self.model(images)

        loss_val = self.criterion(output, target)

        self.val_accuracy(F.softmax(output, dim=1), target)
        self.val_accuracy_macro(F.softmax(output, dim=1), target)
        self.val_accuracy_top5(F.softmax(output, dim=1), target)
        self.val_confusion(F.softmax(output, dim=1), target)

        self.log(
            "val_loss",
            loss_val,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log("val_acc", self.val_accuracy, on_step=False, on_epoch=True)
        self.log("val_acc_macro", self.val_accuracy_macro, on_step=False, on_epoch=True)
        self.log("val_acc_top5", self.val_accuracy_top5, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.params.lr,
            momentum=self.params.momentum,
            weight_decay=self.params.wd,
            nesterov=True,
        )

        steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
        print("steps_per_epoch", steps_per_epoch)

        if self.params.lr_scheduler == "multi_step_lr":
            milestones = np.array([60, 120, 160])
            milestones *= steps_per_epoch
            milestones *= self.params.num_epoch_repetition
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
        elif self.params.lr_scheduler == "step_lr":
            scheduler = lr_scheduler.StepLR(
                optimizer,
                step_size=2 * self.params.num_epoch_repetition * steps_per_epoch,
                gamma=0.9,
            )
        elif self.params.lr_scheduler == "CosineAnnealingLR":
            total_nb_epochs = self.params.num_epoch_repetition * self.params.epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_nb_epochs * steps_per_epoch
            )
        else:
            raise ValueError("unknown lr_scheduler", self.params.lr_scheduler)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


@hydra.main(config_path="configs/cls", config_name="base", version_base="1.1")
def main(cfg: DictConfig):

    cfg.epochs = math.ceil(cfg.epochs)

    if isinstance(cfg.tags, str):
        cfg.tags = [cfg.tags]

    print(OmegaConf.to_yaml(cfg))
    print(HydraConfig.get().job.override_dirname)

    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    wandb_logger = WandbLogger(
        project="chimeramix-classifier",
        tags=cfg.tags,
        save_dir=hydra.utils.to_absolute_path(""),
        config=OmegaConf.to_container(cfg, resolve=True, enum_to_str=True),
    )
    trainer_callbacks = [
        callbacks.LearningRateMonitor(logging_interval="step"),
        ETACallback(epochs=cfg.epochs),
    ]

    if cfg.checkpoint:
        checkpoint_path = checkpoint_cls(
            model_subdir=cfg.model_subdir,
            dataset=cfg.dataset,
            max_labels_per_class=cfg.max_labels_per_class,
            seed=cfg.seed,
            mix=cfg.mix,
            generator_features=cfg.mix_generator_features,
            generator_blocks=cfg.mix_generator_blocks,
            generator_split=cfg.mix_generator_split,
            discriminator_features=cfg.mix_discriminator_features,
            image_size=cfg.mix_generator_image_size,
            mix_mode=cfg.mix_mode,
            mix_size=cfg.mix_size,
            variant=cfg.mix_variant,
        )

        checkpoint_cb = callbacks.ModelCheckpoint(
            dirpath=hydra.utils.to_absolute_path(checkpoint_path),
            save_last=True,
            every_n_epochs=cfg.epochs * cfg.num_epoch_repetition // 2,
        )
        trainer_callbacks.append(checkpoint_cb)
        print("checkpoint", checkpoint_path)

    trainer = pl.Trainer(
        logger=wandb_logger,
        default_root_dir="tmp/lightning_logs",
        gpus=1,
        max_epochs=cfg.epochs * cfg.num_epoch_repetition,
        check_val_every_n_epoch=cfg.num_epoch_repetition,
        callbacks=trainer_callbacks,
        fast_dev_run=cfg.debug,
        enable_checkpointing=cfg.checkpoint,
        precision=16,
    )

    dm_class = get_dm_class(cfg)

    num_dataset_repeats = (
        max(cfg.num_dataset_repeats_base // int(cfg.max_labels_per_class), 1)
        if cfg.max_labels_per_class is not None
        else None
    )

    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    transforms_train, metadata = get_train_transforms(
        dataset=cfg.dataset,
        mean=mean,
        std=std,
        aug_cutout=cfg.augmentation_cutout,
        aug_random_erase=cfg.augmentation_random_erase,
        aug_autoaugment=cfg.augmentation_autoaugment,
        aug_trivialaugment=cfg.augmentation_trivialaugment,
    )
    transforms_test = get_test_transforms(dataset=cfg.dataset, mean=mean, std=std)

    dm = dm_class(
        batch_size=cfg.batch_size,
        max_labels_per_class=cfg.max_labels_per_class,
        max_labels_per_class_seed=cfg.seed,
        num_train_dataset_repeats=num_dataset_repeats,
        pairs_train=True,
        transforms_train=transforms_train,
        transforms_test=transforms_test,
        index_train=True,
        segmentation=cfg.mix_mode == "segmentation",
        data_path=hydra.utils.to_absolute_path("tmp/data"),
        train_workers=8,
        val_workers=8,
    )

    module_mix = None

    if cfg.mix:
        from train_generator import ChimeraMixLightningModel

        checkpoint_path = checkpoint_chimera(
            model_subdir=cfg.model_generator_subdir,
            dataset=cfg.dataset,
            max_labels_per_class=cfg.max_labels_per_class,
            seed=cfg.seed,
            generator_features=cfg.mix_generator_features,
            generator_blocks=cfg.mix_generator_blocks,
            generator_split=cfg.mix_generator_split,
            discriminator_features=cfg.mix_discriminator_features,
            image_size=cfg.mix_generator_image_size,
            mix_mode=cfg.mix_mode,
            mix_size=cfg.mix_size,
            variant=cfg.mix_variant,
            add_file="last.ckpt",
        )
        module_mix = ChimeraMixLightningModel.load_from_checkpoint(
            hydra.utils.to_absolute_path(checkpoint_path)
        )

    model = ClassifierLightningModel(
        image_size=metadata["image_size"],
        num_classes=dm.num_classes,
        mean=mean,
        std=std,
        module_mix=module_mix,
        **cfg,
    )

    trainer.fit(model, datamodule=dm)

    return 0


if __name__ == "__main__":
    atexit.register(lambda: print("\x1b[?25h"))  
    main()
