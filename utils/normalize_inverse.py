from typing import no_type_check
import torch
import torchvision
import numpy as np

class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv

        self.mean_inv = mean_inv
        self.std_inv = std_inv
        mean_inv = torch.as_tensor(mean_inv)
        std_inv = torch.as_tensor(std_inv)
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())

    def transform_np(self, data: np.ndarray):
        data = data[..., :] - self.mean_inv
        data = data[..., :] / self.std_inv
        return data