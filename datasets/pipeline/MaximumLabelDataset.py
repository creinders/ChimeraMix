import numpy as np
import torch
from torch.utils.data.dataset import Dataset, Subset


class MaximumLabelDataset(Subset):
    def __init__(self, ds: Dataset, n, seed=None):
        print("data seed:", seed)

        rng = np.random.default_rng(seed)

        labels = ds.targets

        dataset_idx = []

        max_label = torch.max(labels) if torch.is_tensor(labels) else np.max(labels)

        labels = np.array(labels)

        for i in range(max_label + 1):
            idxs = np.where(labels == i)[0]
            # np.random.shuffle(idxs)
            rng.shuffle(idxs)
            dataset_idx.extend(idxs[:n])
        # print(dataset_idx)
        # np.random.shuffle(dataset_idx)
        rng.shuffle(dataset_idx)
        print("ids", dataset_idx)

        super().__init__(ds, dataset_idx)
        self.targets = labels[dataset_idx]
