import numpy as np
from torch.utils.data.dataset import Dataset, Subset


class RepeatDataset(Subset):
    def __init__(self, ds: Dataset, n, shuffle=True, seed=None):
        print("repeat seed", seed)

        l = len(ds)

        dataset_idx = []

        for _ in range(n):
            ind = list(range(l))
            dataset_idx.extend(ind)

        if shuffle:
            rng = np.random.default_rng(seed)
            # np.random.shuffle(dataset_idx)
            rng.shuffle(dataset_idx)
            # print('repeat shuffled ids:', dataset_idx)

        super().__init__(ds, dataset_idx)
