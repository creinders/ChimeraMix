from datasets.pipeline.DefaultDataset import DefaultDataset
from torch.utils.data.dataset import Dataset


class IndexDataset(DefaultDataset):

    def __init__(self, ds: Dataset) -> None:
        super().__init__()
        self.ds = ds


    def __getitem__(self, idx):
        data_dict = self.ds[idx]
        data_dict['idx']=idx
        return data_dict

                            
