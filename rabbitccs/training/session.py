from collagen.data import DataProvider, ItemLoader

import torch
import numpy as np

from rabbitccs.data.splits import build_splits, parse_item_cb
from rabbitccs.data.transforms import train_test_transforms, estimate_mean_std



def init_seed(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)


def create_data_provider(args, metadata=None):
    if metadata is None:
        metadata = build_splits(args.data_location)
    mean, std = estimate_mean_std(metadata, parse_item_cb, args.num_threads, args.bs)
    item_loaders = dict()
    for stage in ['train', 'val', 'test']:
        item_loaders[f'bfpn_{stage}'] = ItemLoader(meta_data=metadata[stage],
                                                   transform=train_test_transforms(mean, std)[stage],
                                                   parse_item_cb=parse_item_cb,
                                                   batch_size=args.bs, num_workers=args.num_threads,
                                                   shuffle=True if stage == "train" else False)

    return DataProvider(item_loaders)


def parse_multi_label(x, cls, threshold=0.5):
    out = x[:, cls, :, :].unsqueeze(1).gt(threshold)
    return torch.cat((1 - out, out), dim=1).squeeze()


def parse_binary_label(x, threshold=0.5):
    out = x.gt(threshold)
    return torch.cat((1 - out, out), dim=1).squeeze()