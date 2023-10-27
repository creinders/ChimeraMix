from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from typing import Any, Dict, List, Optional
import torch
import numpy as np
from copy import deepcopy
import lightning.pytorch as pl
import time
import datetime

class ETACallback(Callback):

    def __init__(self, epochs) -> None:
        super().__init__()

        self.epochs = epochs
        self.start_time: Optional[float] = None
        self.epoch_start_time: Optional[float] = None
    
    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        self.start_time = time.time()

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.epoch_start_time = time.time()


    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        assert self.start_time is not None
        assert self.epoch_start_time is not None
        now = time.time()
        time_passed = now - self.start_time
        epoch_time = now - self.epoch_start_time
        epoch_time_hours = epoch_time / 60 / 60

        current_epoch = pl_module.current_epoch + 1

        epochs_missing = self.epochs - current_epoch
        eta_seconds = time_passed / current_epoch * epochs_missing
        eta_hours = eta_seconds / 60 / 60

        pl_module.log('eta_hours', eta_hours, on_epoch=True, prog_bar=True)
        pl_module.log('time_per_epoch_hours', epoch_time_hours, on_epoch=True, prog_bar=False)
        # max_memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        # pl_module.log('max_mem (MB)', max_memory, on_epoch=False, prog_bar=True)
