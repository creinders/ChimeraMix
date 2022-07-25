from torch.utils.data.dataset import Dataset


## implements some defaults for the Datasets
class DefaultDataset(Dataset):
    def __len__(self):
        return len(self.ds)

    def __getattr__(self, item_name):
        if item_name == "targets":
            return self.ds.targets
