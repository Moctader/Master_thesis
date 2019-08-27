import torch
import numpy as np
from functools import partial

import solt.transforms as slt
import solt.data as sld
import solt.core as slc

from collagen.data.utils import ApplyTransform, Compose
from collagen.data import ItemLoader


def normalize_channel_wise(tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Normalizes given tensor channel-wise
    Parameters
    ----------
    tensor: torch.Tensor
        Tensor to be normalized
    mean: torch.tensor
        Mean to be subtracted
    std: torch.Tensor
        Std to be divided by
    Returns
    -------
    result: torch.Tensor
    """
    if len(tensor.size()) != 3:
        raise ValueError
    # Original version
    """
    for channel in range(tensor.size(0)):
        tensor[channel, :, :] -= mean[channel]
        tensor[channel, :, :] /= std[channel]

    return tensor
    """
    # Modified shape
    for channel in range(tensor.size(2)):
        tensor[:, :, channel] -= mean[channel]
        tensor[:, :, channel] /= std[channel]

    return tensor


def numpy2tens(x: np.ndarray, dtype='f') -> torch.Tensor:
    """Converts a numpy array into torch.Tensor
    Parameters
    ----------
    x: np.ndarray
        Array to be converted
    dtype: str
        Target data type of the tensor. Can be f - float and l - long
    Returns
    -------
    result: torch.Tensor
    """
    x = x.squeeze()
    x = torch.from_numpy(x)
    if x.dim() == 2:  # CxHxW format
        x = x.unsqueeze(0)

    if dtype == 'f':
        return x.float()
    elif dtype == 'l':
        return x.long()
    else:
        raise NotImplementedError


def wrap_solt(entry):
    return sld.DataContainer(entry, 'IM')


def unwrap_solt(dc):
    return dc.data


def train_test_transforms(conf, mean=None, std=None, crop_size=(512, 1024)):
    prob = conf['training']['transform_probability']
    transforms = [
        slt.RandomFlip(p=prob),
        slt.RandomRotate(rotation_range=(conf['training']['rotation_min'], conf['training']['rotation_max']), p=prob),
        slt.RandomScale(range_x=(conf['training']['scale_min'], conf['training']['scale_max']),
                        range_y=(conf['training']['scale_min'], conf['training']['scale_max']), same=False, p=prob),
        slt.PadTransform(pad_to=crop_size[1]),
        slt.CropTransform(crop_mode='r', crop_size=crop_size),
        slt.ImageGammaCorrection(gamma_range=(conf['training']['gamma_min'], conf['training']['gamma_max']), p=prob)
    ]

    train_trf = [
        wrap_solt,
        slc.Stream(transforms),
        unwrap_solt,
        ApplyTransform(numpy2tens, (0, 1, 2))
    ]

    val_trf = [
        wrap_solt,
        slc.Stream([
            slt.PadTransform(pad_to=crop_size[1]),
            slt.CropTransform(crop_mode='r', crop_size=crop_size)
        ]),
        unwrap_solt,
        ApplyTransform(numpy2tens, idx=(0, 1, 2))
    ]

    if mean is not None and std is not None:
        train_trf.append(ApplyTransform(partial(normalize_channel_wise, mean=mean, std=std)))

    if mean is not None and std is not None:
        val_trf.append(ApplyTransform(partial(normalize_channel_wise, mean=mean, std=std)))

    train_trf = Compose(train_trf)
    val_trf = Compose(val_trf)

    return {'train': train_trf, 'val': val_trf, 'test': val_trf}


def estimate_mean_std(config, metadata, parse_item_cb, num_threads=8, bs=16):
    mean_std_loader = ItemLoader(meta_data=metadata,
                                 transform=train_test_transforms(config)['train'],
                                 parse_item_cb=parse_item_cb,
                                 batch_size=bs, num_workers=num_threads,
                                 shuffle=False)

    mean = None
    std = None
    for i in range(len(mean_std_loader)):
        for batch in mean_std_loader.sample():
            if mean is None:
                mean = torch.zeros(batch['data'].size(1))
                std = torch.zeros(batch['data'].size(1))
            # for channel in range(batch['data'].size(1)):
            #     mean[channel] += batch['data'][:, channel, :, :].mean().item()
            #     std[channel] += batch['data'][:, channel, :, :].std().item()
            mean += batch['data'].mean().item()
            std += batch['data'].std().item()

    mean /= len(mean_std_loader)
    std /= len(mean_std_loader)

    return mean, std