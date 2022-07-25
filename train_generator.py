import random
from typing import Optional

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
from torchinfo import summary
from torchvision.utils import make_grid

from models.chimera import ChimeraModel
from models.discriminator import Discriminator
from utils.checkpoints import check_checkpoint_exists, checkpoint_chimera
from datasets.utils import get_dm_class
from utils.eta_callback import ETACallback
from utils.lap_pyramid_loss import LapLoss
from utils.normalize_inverse import NormalizeInverse
from utils.tensor_utils import padding_tensor
from utils.transforms import get_test_transforms, get_train_transforms


class ChimeraMixLightningModel(LightningModule):
    def __init__(self, image_size, num_classes, mean, std, **hparams):
        super().__init__()

        self.params = DictConfig(hparams)
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.mean = mean
        self.std = std
        self.image_size = image_size

        self.generator = ChimeraModel(
            nc=3,
            num_residual_blocks=self.params.generator_blocks,
            split_index=self.params.generator_split,
            image_size=self.image_size,
            ngf=self.params.generator_features,
        )

        self.discriminator = Discriminator(
            input_shape=(3, self.image_size, self.image_size),
            ndf=self.params.discriminator_features,
        )

        self.noise_a = torch.ones((1, self.image_size // 4, self.image_size // 4))
        self.noise_b = torch.zeros((1, self.image_size // 4, self.image_size // 4))
        noise_shape = (1, 2, self.image_size // 4, self.image_size // 4)

        summary(
            self.generator,
            [
                (1, 3, self.image_size, self.image_size),
                (1, 3, self.image_size, self.image_size),
                noise_shape,
            ],
            depth=5,
        )
        summary(
            self.discriminator,
            [(1, 3, self.image_size, self.image_size), (1, num_classes)],
        )

        self.criterion_perceptual = LapLoss()
        self.normalize_inv = NormalizeInverse(self.mean, self.std)
        self.criterion_GAN = nn.MSELoss()
        self.log_next_train_batch_cls = False
        self.log_next_train_batch_gen = False

    def setup(self, stage: Optional[str] = None) -> None:

        if hasattr(self.trainer.datamodule, "classes"):
            classes = self.trainer.datamodule.classes
        else:
            classes = list([str(i) for i in range(self.num_classes)])
        classes = list(classes)
        classes.append("background")
        self.index_to_class = {i: c for i, c in enumerate(classes)}

    def gen_noise_a(self, bs):
        f = self.noise_a.unsqueeze(0).repeat(bs, 1, 1, 1).to(self.device)
        return torch.cat((f, 1 - f), dim=1)

    def gen_noise_b(self, bs):
        f = self.noise_b.unsqueeze(0).repeat(bs, 1, 1, 1).to(self.device)
        return torch.cat((f, 1 - f), dim=1)

    def gen_noise(
        self,
        bs,
        segmentation_a: Optional[torch.Tensor] = None,
        segmentation_b: Optional[torch.Tensor] = None,
        resize=True,
    ):
        if self.params.mix_mode == "segmentation":
            assert isinstance(segmentation_b, torch.Tensor)
            segmentation_b = segmentation_b.to(self.device)
            num_segments = segmentation_b.max(dim=-1)[0].max(-1)[0] + 1
            lower_bounds = [
                torch.randint(max(1, n_seg.item() - 1), size=[], device=self.device)
                for n_seg in num_segments
            ]
            upper_bounds = [
                torch.randint(
                    l + 1, max(l + 2, n_seg.item()), size=[], device=self.device
                )
                for l, n_seg in zip(lower_bounds, num_segments)
            ]
            sampled_segments = [
                torch.arange(l, u, device=self.device)
                for l, u in zip(lower_bounds, upper_bounds)
            ]
            sampled_segments_padded = padding_tensor(sampled_segments, pad_value=-1)[
                0
            ].to(self.device)
            ones = torch.ones_like(segmentation_b)
            zeros = torch.zeros_like(segmentation_b)
            mask = torch.where(
                (
                    segmentation_b[..., None]
                    == sampled_segments_padded[:, None, None, ...]
                ).any(-1),
                ones,
                zeros,
            ).float()
            f = mask[:, None, ...]
            if resize:
                target_size = self.image_size // 4
                f = F.interpolate(
                    f,
                    size=(target_size, target_size),
                    mode=self.params.noise_interpolation_mode,
                )
            return torch.cat((f, 1 - f), dim=1)
        else:
            base_noise_size = self.params.mix_size

            f = torch.randint(
                0,
                2,
                size=(bs, 1, base_noise_size, base_noise_size),
                device=self.device,
            ).float()

            if resize:
                target_size = self.image_size // 4
            else:
                target_size = self.image_size
            pad = target_size // base_noise_size
            scale_size = target_size + pad
            f = F.interpolate(
                f,
                size=(scale_size, scale_size),
                mode=self.params.noise_interpolation_mode,
            )
            x = random.randint(0, pad - 1)
            y = random.randint(0, pad - 1)
            f = f[..., y : y + target_size, x : x + target_size]

            return torch.cat((f, 1 - f), dim=1)

    def forward(self, a, b, f):
        return self.generator(a, b, f)

    def on_epoch_start(self) -> None:

        if self.current_epoch % max(self.params.num_epoch_repetition, 5) == 0:
            self.log_next_train_batch_cls = True
            self.log_next_train_batch_gen = True

    def training_step(self, batch, batch_idx, optimizer_idx):
        data_a, data_b = batch
        images_a = data_a["image"]
        images_b = data_b["image"]
        target_a = data_a["target"].long()
        target_b = data_b["target"].long()

        if self.params.mix_mode == "segmentation":
            segmentation_a = data_a["segmentation"]
            segmentation_b = data_b["segmentation"]
        else:
            segmentation_a, segmentation_b = None, None

        bs, _, height, width = images_a.shape
        real_label = torch.tensor(1, dtype=torch.float32, device=self.device)
        fake_label = torch.tensor(0, dtype=torch.float32, device=self.device)

        if optimizer_idx == 0:
            target_a_oh = F.one_hot(target_a, num_classes=self.num_classes).float()

            mix_mask_a = self.gen_noise_a(bs)
            mix_mask_b = self.gen_noise_b(bs)

            f_a = self.generator.encode(images_a)
            f_b = self.generator.encode(images_b)

            if self.params.mix_mode == "segmentation":
                mix_mask = self.gen_noise(
                    bs, segmentation_a=segmentation_a, segmentation_b=segmentation_b
                )
            else:
                mix_mask = self.gen_noise(bs)

            gen_a, gen_a_a, gen_a_b, ms_a = self.generator.mixDecode(
                f_a, f_b, mix_mask_a
            )
            gen_b, gen_b_a, gen_b_b, ms_b = self.generator.mixDecode(
                f_a, f_b, mix_mask_b
            )
            gen, gen_g_a, gen_g_b, ms_g = self.generator.mixDecode(f_a, f_b, mix_mask)

            rec_loss_a = F.mse_loss(gen_a, images_a, reduction="mean")
            rec_loss_b = F.mse_loss(gen_b, images_b, reduction="mean")
            rec_loss = rec_loss_a + rec_loss_b
            loss = 1000 * rec_loss

            self.log("train_gen_rec", rec_loss, on_step=True, on_epoch=True)
            self.log("train_gen_rec_a", rec_loss_a, on_step=True, on_epoch=True)
            self.log("train_gen_rec_b", rec_loss_b, on_step=True, on_epoch=True)

            disc_output = self.discriminator(gen, target_a_oh)

            errGeneratorDisc = self.criterion_GAN(
                disc_output, real_label.expand_as(disc_output)
            )
            loss += errGeneratorDisc
            self.log("train_gen_disc", errGeneratorDisc, on_step=True, on_epoch=True)

            if self.criterion_perceptual is not None:
                loss_perceptual_a = self.criterion_perceptual(gen_a, images_a)
                loss_perceptual_b = self.criterion_perceptual(gen_b, images_b)
                loss_perceptual = loss_perceptual_a + loss_perceptual_b

                self.log(
                    "train_loss_perceptual_a",
                    loss_perceptual_a,
                    on_step=True,
                    on_epoch=True,
                )
                self.log(
                    "train_loss_perceptual_b",
                    loss_perceptual_b,
                    on_step=True,
                    on_epoch=True,
                )
                self.log(
                    "train_loss_perceptual",
                    loss_perceptual,
                    on_step=True,
                    on_epoch=True,
                )

                loss += loss_perceptual

            self.log("train_gen_loss_total", loss, on_step=True, on_epoch=True)

            if self.log_next_train_batch_gen:
                n = min(images_a.size(0), 8)

                images_a = images_a[:n]
                images_b = images_b[:n]
                gen_a = gen_a[:n]
                gen_b = gen_b[:n]
                gen = gen[:n]

                data = torch.cat((images_a, gen_a, images_b, gen_b, gen), dim=0)
                data = self.normalize_inv(data)

                mix_mask = torch.split(mix_mask, 1, dim=1)
                mix_mask = torch.cat(mix_mask, dim=2)

                mix_mask = mix_mask[:n]

                self.logger.experiment.log(
                    {
                        "generated": wandb.Image(
                            make_grid(data.float(), normalize=False, nrow=n,),
                            mode="RGB",
                        ),
                        "mask": wandb.Image(
                            make_grid(mix_mask.float(), normalize=False, nrow=1,),
                        ),
                    },
                )

                if self.params.mix_mode == "segmentation":
                    segmentation_a = segmentation_a[:n]
                    segmentation_b = segmentation_b[:n]
                    self.visualize_sampling(
                        images_a, images_b, segmentation_a, segmentation_b
                    )
                else:
                    self.visualize_sampling(images_a, images_b)

                self.log_next_train_batch_gen = False

        elif optimizer_idx == 1:

            target_a_oh = F.one_hot(target_a, num_classes=self.num_classes).float()

            f_a = self.generator.encode(images_a)
            f_b = self.generator.encode(images_b)

            if self.params.mix_mode == "segmentation":
                mask = self.gen_noise(bs, segmentation_a, segmentation_b)
            else:
                mask = self.gen_noise(bs)

            gen, gen_a, gen_b, ms_g = self.generator.mixDecode(f_a, f_b, mask)

            fake_input = gen

            real_images, i = (
                (images_a, 0) if np.random.uniform() < 0.5 else (images_b, 1)
            )

            output_real = self.discriminator(real_images, target_a_oh)

            errRefinerD_real = self.criterion_GAN(
                output_real, real_label.expand_as(output_real)
            )
            output_fake = self.discriminator(fake_input.detach(), target_a_oh)
            errRefinerD_fake = self.criterion_GAN(
                output_fake, fake_label.expand_as(output_fake)
            )

            errRefinerD = errRefinerD_real + errRefinerD_fake
            loss = errRefinerD

            self.log(
                "errDisc", errRefinerD, on_step=True, on_epoch=True, prog_bar=False
            )

        else:
            raise ValueError("unknown optimizer index", optimizer_idx)
        return loss

    def visualize_sampling(
        self, images_a, images_b, segmentation_a=None, segmentation_b=None,
    ):

        bs = images_a.size(0)
        assert bs == images_b.size(0)

        n = min(bs, 8)
        noise_samples = 6
        all_samplings_combined = []

        for i in range(n):
            image_a_i = images_a[i]
            image_b_i = images_b[i]
            image_a_i_batch = image_a_i.unsqueeze(0).repeat(noise_samples, 1, 1, 1)
            image_b_i_batch = image_b_i.unsqueeze(0).repeat(noise_samples, 1, 1, 1)

            if self.params.mix_mode == "segmentation":
                image_a_i_segmentation = segmentation_a[i].unsqueeze(0).repeat(noise_samples, 1, 1)
                image_b_i_segmentation = segmentation_b[i].unsqueeze(0).repeat(noise_samples, 1, 1)
                mask = self.gen_noise(noise_samples, image_a_i_segmentation, image_b_i_segmentation)
            else:
                mask = self.gen_noise(noise_samples)

            mask = mask.to(images_a.device)
            samplings = [image_a_i, image_b_i]

            gen, gen_a, gen_b, ms_g = self.generator(
                image_a_i_batch, image_b_i_batch, mask
            )

            for x in torch.split(gen, 1):
                samplings.append(x.squeeze(0))
            samplings_combined = torch.concat(samplings, dim=2)
            all_samplings_combined.append(samplings_combined)

        all_samplings_combined = torch.concat(all_samplings_combined, dim=1)
        all_samplings_combined = self.normalize_inv(all_samplings_combined)
        self.logger.experiment.log(
            {"sampling": wandb.Image(all_samplings_combined, mode="RGB",)}, commit=False
        )

    def configure_optimizers(self):

        optimizer_g = optim.Adam(
            self.generator.parameters(), lr=self.params.lr, betas=(0.5, 0.999)
        )
        optimizer_d = optim.Adam(
            self.discriminator.parameters(), lr=self.params.lr, betas=(0.5, 0.999)
        )

        steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
        print("steps_per_epoch", steps_per_epoch)

        if self.params.lr_scheduler == "multi_step_lr":
            milestones = np.array([60, 120, 160])
            milestones *= steps_per_epoch  # Workaround because interval epoch not working with check_val_every_n_epoch
            milestones *= self.params.num_epoch_repetition

            scheduler_g = lr_scheduler.MultiStepLR(optimizer_g, milestones, gamma=0.2)
            scheduler_d = lr_scheduler.MultiStepLR(optimizer_d, milestones, gamma=0.2)
        else:
            raise ValueError("unknown lr_scheduler", self.params.lr_scheduler)

        return [
            {
                "optimizer": optimizer_g,
                "lr_scheduler": {"scheduler": scheduler_g, "interval": "step"},
                "frequency": None,
            },
            {
                "optimizer": optimizer_d,
                "lr_scheduler": {"scheduler": scheduler_d, "interval": "step"},
                "frequency": None,
            },
        ]


@hydra.main(config_path="configs/gen", config_name="base", version_base="1.1")
def main(cfg: DictConfig):

    if isinstance(cfg.tags, (list, tuple)):
        cfg.tags = [cfg.tags]

    print(OmegaConf.to_yaml(cfg))
    print(HydraConfig.get().job.override_dirname)

    project_name = "chimeramix-generator"

    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    wandb_logger = WandbLogger(
        project=project_name,
        tags=cfg.tags,
        save_dir=hydra.utils.to_absolute_path(""),
        config=OmegaConf.to_container(cfg, resolve=True, enum_to_str=True),
    )

    checkpoint_path = checkpoint_chimera(
        model_subdir=cfg.model_subdir,
        dataset=cfg.dataset,
        max_labels_per_class=cfg.max_labels_per_class,
        seed=cfg.seed,
        generator_features=cfg.generator_features,
        generator_blocks=cfg.generator_blocks,
        generator_split=cfg.generator_split,
        discriminator_features=cfg.discriminator_features,
        image_size=cfg.generator_image_size,
        mix_mode=cfg.mix_mode,
        mix_size=cfg.mix_size,
        variant=cfg.variant,
    )

    if not cfg.override and check_checkpoint_exists(
        checkpoint_path=checkpoint_path, epochs=cfg.epochs * cfg.num_epoch_repetition
    ):
        print("experiment already existing. Skip")
        return 0

    trainer_callbacks = [
        callbacks.LearningRateMonitor(logging_interval="step"),
        ETACallback(epochs=cfg.epochs),
    ]

    if cfg.checkpoint:
        checkpoint_cb = callbacks.ModelCheckpoint(
            dirpath=hydra.utils.to_absolute_path(checkpoint_path),
            save_last=True,
            every_n_epochs=cfg.epochs * cfg.num_epoch_repetition // 2,
        )
        trainer_callbacks.append(checkpoint_cb)

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
        dataset=cfg.dataset, mean=mean, std=std, image_size=cfg.generator_image_size
    )
    transforms_test = get_test_transforms(
        dataset=cfg.dataset, mean=mean, std=std, image_size=cfg.generator_image_size
    )

    dm = dm_class(
        batch_size=cfg.batch_size,
        max_labels_per_class=cfg.max_labels_per_class,
        max_labels_per_class_seed=cfg.seed,
        num_train_dataset_repeats=num_dataset_repeats,
        pairs_train=True,
        transforms_train=transforms_train,
        transforms_test=transforms_test,
        index_train=True,
        train_workers=8 if not cfg.debug else 0,
        data_path=hydra.utils.to_absolute_path("tmp/data"),
        segmentation=cfg.mix_mode == "segmentation",
    )

    model = ChimeraMixLightningModel(
        image_size=metadata["image_size"],
        num_classes=dm.num_classes,
        mean=mean,
        std=std,
        checkpoint_path=checkpoint_path,
        **cfg,
    )

    trainer.fit(model, datamodule=dm)

    return 0


if __name__ == "__main__":
    main()
