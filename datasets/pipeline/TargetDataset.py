import warnings

from datasets.pipeline.DefaultDataset import DefaultDataset
from torch import cat
from torch.utils.data.dataset import Dataset


class TargetDataset(DefaultDataset):
    def __init__(self, ds: Dataset, label_index=None) -> None:
        super().__init__()
        self.ds = ds

        if hasattr(ds, "targets") and getattr(ds, "targets") is not None:
            self.targets = getattr(ds, "targets")
            self.label_index = None
        elif hasattr(ds, "labels") and getattr(ds, "labels") is not None:
            self.targets = getattr(ds, "labels")
            self.label_index = None
        else:
            if label_index is None:
                warnings.warn(
                    "dataset has no attribute >targets< and no index for target column provided --> using default: -1"
                )
                self.label_index = -1
            else:
                self.label_index = label_index

            self.targets = [ds[i][self.label_index] for i in range(len(ds))]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        data = self.ds[idx]
        if isinstance(data, dict):
            return data

        if self.label_index is None:
            if isinstance(data, (list, tuple)):
                assert len(data) == 2
                return {"image": self.ds[idx][0], "target": self.targets[idx]}
            else:
                return {"image": self.ds[idx], "target": self.targets[idx]}
        else:  # remvove target from data, otherwise its redundant

            skip_dim_index = (
                len(data) + self.label_index
                if self.label_index < 0
                else self.label_index
            )  # dealing with negative indexing
            data = [data[x] for x in range(len(data)) if x != skip_dim_index]

            image = cat(data)
            return {"image": image, "target": self.targets[idx]}

