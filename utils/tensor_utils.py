import torch
import torch.nn.functional as F

def padding_tensor(sequences, pad_value=0):
    """
    :param sequences: list of tensors
    :return:
    """
    num = len(sequences)
    max_len = max([s.size(0) for s in sequences])
    out_dims = (num, max_len)
    out_tensor = sequences[0].data.new(*out_dims).fill_(pad_value)
    mask = sequences[0].data.new(*out_dims).fill_(0)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        out_tensor[i, :length] = tensor
        mask[i, :length] = 1
    return out_tensor, mask


def linear_interpolation(a, b, f):

    if torch.is_tensor(f):
        assert f.size(0) == 1 or f.size(0) == a.size(0)
        # f = f.expand_as(a)

        for _ in range(len(a.size()) - len(f.size())):
            f = f.unsqueeze(-1)

    return (1 - f) * a + f * b


def resize_to(source, target_size, mode="bilinear", add_channel=False):
    if not isinstance(target_size, (list, tuple)):
        target_size = (target_size, target_size)
    source_size = tuple(source.size()[-2:])
    # target_size = target.size()[-2:]
    if source_size != target_size:

        if add_channel:
            source = source.unsqueeze(1)

        align_corners = True if mode not in ("nearest", "area") else None
        r = F.interpolate(
            source, size=target_size, align_corners=align_corners, mode=mode
        )
        # r = kornia.geometry.transform.resize(source, size=target_size, align_corners=align_corners, interpolation=mode, antialias=True)
        if add_channel:
            r = r.squeeze(1)

        return r
    else:
        return source
