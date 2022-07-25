import random

import numpy as np
from datasets.pipeline.DefaultDataset import DefaultDataset
from torch.utils.data.dataset import Dataset


class PairDataset(DefaultDataset):
    def __init__(
        self, ds: Dataset, same_class=True, include_self=True, n=1, label_index=-1
    ):
        super().__init__()

        self.ds = ds
        self.same_class = same_class
        self.n = n
        self.label_index = label_index
        self.include_self = include_self

        if same_class:

            labels = ds.targets

            labels = np.array(labels)
            num_classes = np.max(labels) + 1

            label2idx = [[] for _ in range(num_classes)]
            for i in range(labels.shape[0]):
                label = labels[i]
                label2idx[label].append(i)

            self.label2idx = label2idx

    def sample_indices_same_label(self, label, current_index):
        indices = self.label2idx[label]

        if not self.include_self:
            indices = [i for i in indices if i != current_index]

        return random.sample(indices, self.n)

    def __getitem__(self, idx):

        data_dict = self.ds[idx]
        label = data_dict["target"]

        if self.same_class:
            knn_indices = self.sample_indices_same_label(label, idx)
        else:
            knn_indices = random.sample(list(range(len(self.ds))), self.n)

        result = [data_dict]

        for i in range(len(knn_indices)):
            data_knn = self.ds[knn_indices[i]]
            result.append(data_knn)

        return result

