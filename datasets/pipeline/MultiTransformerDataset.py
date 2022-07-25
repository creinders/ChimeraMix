import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from datasets.pipeline.DefaultDataset import DefaultDataset
from torch import is_tensor, rand
from torch.utils.data.dataset import Dataset


class MultiTransformerDataset(DefaultDataset):
    def __init__(self, ds: Dataset, transformers) -> None:
        super().__init__()
        self.ds = ds
        self.transformers = transformers

    def __getitem__(self, idx):
        data_dict = self.ds[idx]

        if self.transformers is None:
            pass

        elif isinstance(self.transformers, (list, tuple)):
            data_dict = [t(data_dict) for t in self.transformers]

        else:  # only one
            data_dict = self.transformers(data_dict)

        return data_dict

    def __len__(self):
        return len(self.ds)

    def apply_gt_transformations(self, data_dict):

        img, gt_seg = data_dict["data"], data_dict["segmentation"]
        if is_tensor(gt_seg):
            gt_seg = gt_seg.unsqueeze(0) if len(gt_seg.shape) == 2 else gt_seg
        if isinstance(self.transformers, transforms.Compose):
            transform_seq = self.transformers.transforms
        elif isinstance(self.transformers, (tuple, list)):
            transform_seq = self.transformers
        else:  # single transform
            transform_seq = [self.transformers]

        for t in transform_seq:
            if isinstance(t, transforms.RandomResizedCrop):
                args = t.get_params(img, t.size, t.ratio)
                img = F.resized_crop(img, *args, t.size, t.interpolation)
                gt_seg = F.resized_crop(gt_seg, *args, t.size, t.interpolation)

            elif isinstance(t, transforms.RandomHorizontalFlip):
                if rand(1) < t.p:
                    img = F.hflip(img)
                    gt_seg = F.hflip(gt_seg)

            elif isinstance(t, transforms.ToTensor):
                img = t(img) if not is_tensor(img) else img
                gt_seg = t(gt_seg) if not is_tensor(gt_seg) else gt_seg

            elif isinstance(t, transforms.Normalize):
                # modify image only
                img = t(img)

            elif isinstance(t, (transforms.Resize, transforms.CenterCrop)):
                img, gt_seg = t(img), t(gt_seg)

            else:
                raise NotImplementedError(
                    "make sure you provide an implementation in this function that modifies the seg-mask accordingly"
                )

        data_dict["data"] = img
        gt_seg = gt_seg.squeeze(0) if len(gt_seg.shape) == 3 else gt_seg
        data_dict["segmentation"] = gt_seg
        return data_dict

