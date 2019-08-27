import pathlib
import argparse
import yaml
import torch
import numpy as np
import time
import socket
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from functools import partial

from collagen.data import DataProvider, ItemLoader
from collagen.core.utils import auto_detect_device
from collagen.callbacks.meters import RunningAverageMeter, JaccardDiceMeter, AccuracyMeter, ItemWiseBinaryJaccardDiceMeter
from collagen.callbacks.logging import ScalarMeterLogger
from collagen.callbacks import ModelSaver, ImageMaskVisualizer, SimpleLRScheduler

from rabbitccs.data.splits import build_splits, parse_item_cb
from rabbitccs.data.transforms import train_test_transforms, estimate_mean_std


def init_experiment():
    # Input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_location', type=pathlib.Path, default='../../../Data')
    parser.add_argument('--workdir', type=pathlib.Path, default='../../../workdir/')
    parser.add_argument('--experiment', default='./experiment_config.yml')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_threads', type=int, default=16)
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--crop_size', type=tuple, default=(512, 1024))
    args = parser.parse_args()
    with open(args.experiment, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Seeding
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Initialize working directories
    snapshots_dir = pathlib.Path(args.workdir) / 'snapshots'
    logs_dir = pathlib.Path(args.workdir) / 'logs'
    snapshots_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    device = auto_detect_device()

    snapshot_name = time.strftime(f'{socket.gethostname()}_%Y_%m_%d_%H_%M_%S')
    (snapshots_dir / snapshot_name).mkdir(exist_ok=True, parents=True)

    return args, config, device, snapshots_dir, snapshot_name, logs_dir


def init_callbacks(fold_id, config, snapshots_dir, snapshot_name, model, optimizer, data_provider, mean, std):
    # Snapshot directory
    current_snapshot_dir = snapshots_dir / snapshot_name
    crop = config['training']['crop_size']
    log_dir = current_snapshot_dir / f"{crop[0]}x{crop[1]}_fold_{fold_id}_logs"
    device = next(model.parameters()).device
    # Tensorboard
    writer = SummaryWriter(comment='RabbitCCS', log_dir=log_dir)
    prefix = f"{crop[0]}x{crop[1]}_fold_{fold_id}"
    threshold = 0.3 if config['training']['log_jaccard'] else 0.5
    # Callbacks
    train_cbs = (RunningAverageMeter(prefix="train", name="loss"),
                 ScalarMeterLogger(writer, comment='training', log_dir=str(log_dir))
                 )

    val_cbs = (RunningAverageMeter(prefix="eval", name="loss"),
               ImageMaskVisualizer(writer, data_provider, log_dir=str(log_dir), comment='visualize',
                                   mean=mean, std=std),
               ModelSaver(metric_names='eval/loss',
                          prefix=prefix,
                          save_dir=str(current_snapshot_dir),
                          conditions='min', model=model),
               ItemWiseBinaryJaccardDiceMeter(prefix="eval", name='jaccard',
                                              parse_output=partial(parse_binary_label, threshold=threshold),
                                              parse_target=lambda x: x.squeeze().to(device)),
               ItemWiseBinaryJaccardDiceMeter(prefix="eval", name='dice',
                                              parse_output=partial(parse_binary_label, threshold=threshold),
                                              parse_target=lambda x: x.squeeze().to(device)),
               # Reduce LR on plateau
               SimpleLRScheduler('eval/loss', ReduceLROnPlateau(optimizer,
                                                                patience=config['training']['patience'],
                                                                factor=config['training']['factor'],
                                                                eps=config['training']['eps'])),
               ScalarMeterLogger(writer=writer, comment='validation', log_dir=log_dir))

    return train_cbs, val_cbs


def create_data_provider(args, config, metadata=None, mean=None, std=None):
    if (metadata or mean or std) is None:
        metadata = build_splits(args.data_location)
        mean, std = estimate_mean_std(config, metadata['train'], parse_item_cb, args.num_threads, args.bs)
    item_loaders = dict()
    for stage in ['train', 'val', 'test']:
        item_loaders[f'bfpn_{stage}'] = ItemLoader(meta_data=metadata[stage],
                                                   transform=train_test_transforms(config, mean, std, crop_size=args.crop_size)[stage],
                                                   parse_item_cb=parse_item_cb,
                                                   batch_size=args.bs, num_workers=args.num_threads,
                                                   shuffle=True if stage == "train" else False)

    return DataProvider(item_loaders)


def parse_multi_label(x, cls, threshold=0.5):
    out = x[:, cls, :, :].unsqueeze(1).gt(threshold)
    return torch.cat((1 - out, out), dim=1).squeeze()


def parse_binary_label(x, threshold=0.5):
    out = x.gt(threshold)
    #return torch.cat((~out, out), dim=1).squeeze().float()
    return out.squeeze().float()