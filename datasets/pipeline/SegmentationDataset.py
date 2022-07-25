import numpy as np
import skimage.segmentation as seg
from datasets.pipeline.DefaultDataset import DefaultDataset
from torch.utils.data.dataset import Dataset


class SegmentationDataset(DefaultDataset):
    def __init__(self, dataset: Dataset, method: str, method_kwargs):
        super().__init__()
        self.ds = dataset
        self.method = seg.__dict__[method]
        self.method_kwargs = method_kwargs

    def __getitem__(self, idx):
        data_dict = self.ds[idx]
        image = data_dict["image"]
        segmentation = self.method(
            np.array(image).transpose((1, 2, 0)), **self.method_kwargs
        ).astype(np.uint8)
        data_dict["segmentation"] = segmentation
        return data_dict

