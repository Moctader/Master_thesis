import pathlib
import argparse
import yaml
import torch.nn as nn
# TODO install packages
import numpy as np
import time
import socket
import dill
import json
import cv2
import matplotlib.pyplot as plt
import solt.data as sld
#from tensorboardX import SummaryWriter
import torch
#import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
#from HMDS_collagen.hmdscollagen.data import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
import torch.nn.functional as F
import pandas as pd


from collagen.data import DataProvider, ItemLoader
from collagen.core.utils import auto_detect_device
from collagen.callbacks.meters import Meter
from collagen.callbacks.meters import RunningAverageMeter
from collagen.callbacks.meters import ItemWiseBinaryJaccardDiceMeter
from collagen.callbacks.logging import ScalarMeterLogger
from collagen.callbacks.train import ModelSaver
from collagen.callbacks.logging import ImageMaskVisualizer
from collagen.callbacks.lr_scheduling import SimpleLRScheduler
from collagen.losses.segmentation import CombinedLoss, BCEWithLogitsLoss2d, SoftJaccardLoss
from hmdscollagen.data.transforms import train_test_transforms, estimate_mean_std
#from hmdscollagen.training.models import SimpleNet
from hmdscollagen.training.net import ReconNet
from scipy import ndimage, misc
#from splits import metadata
from functools import partial

def init_experiment(experiment='2D'):
    # Input arguments
    # TODO check paths and input variables
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_location', type=pathlib.Path, default='/data/Repositories/HMDS_orientation/Data/train_rotated/')
    parser.add_argument('--workdir', type=pathlib.Path, default='/data/Repositories/HMDS_collagen/workdir/')
    parser.add_argument('--experiment', default='/data/Repositories/HMDS_collagen/experiments/experiment_config_HMDS.yml')
    parser.add_argument('--data_dir', default="/data/Repositories/HMDS_orientation/Data/train_rotated")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_unet', type=bool, default=False)
    parser.add_argument('--num_threads', type=int, default=2)  # parallel processing
    parser.add_argument('--bs', type=int, default=32)  # images per batch (batch size)
    parser.add_argument('--n_epochs', type=int, default=1)  # iteration for training

    args = parser.parse_args()

    # Open configuration file
    with open(args.experiment, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Seeding
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Initialize working directories
    snapshots_dir = pathlib.Path(args.workdir) / 'snapshots'
    snapshots_dir.mkdir(exist_ok=True)
    device = auto_detect_device()
    snapshot_name = time.strftime(f'{socket.gethostname()}_%Y_%m_%d_%H_%M_%S')
    (snapshots_dir / snapshot_name).mkdir(exist_ok=True, parents=True)

    # Save the experiment parameters
    with open(snapshots_dir / snapshot_name / 'config.yml', 'w') as f:
        yaml.dump(config, f, Dumper=yaml.Dumper, default_flow_style=False)
    # args
    with open(snapshots_dir / snapshot_name / 'args.dill', 'wb') as f:
        dill.dump(args, f)

    return args, config, device, snapshots_dir, snapshot_name


def init_callbacks(fold_id, config, snapshots_dir, snapshot_name, model, optimizer, data_provider, mean, std):
    # Snapshot directory
    current_snapshot_dir = snapshots_dir / snapshot_name
    crop = config['training']['crop_size']
    log_dir = current_snapshot_dir / f"fold_{fold_id}_log"
    device = next(model.parameters()).device

    # Tensorboard
    writer = SummaryWriter(comment='RabbitCCS', log_dir=log_dir, flush_secs=15, max_queue=1)
    prefix = f"{crop[0]}x{crop[1]}_fold_{fold_id}"

    # Set threshold
    threshold = 0.3 if config['training']['log_jaccard'] else 0.5

    # Callbacks
    train_cbs = (RunningAverageMeter(prefix="train", name="loss"),
                 ScalarMeterLogger(writer, comment='training', log_dir=str(log_dir))
                 )

    val_cbs = (RunningAverageMeter(prefix="eval", name="loss"),
              # ImageMaskVisualizer(writer, log_dir=str(log_dir), comment='visualize', mean=mean, std=std),
               ModelSaver(metric_names='eval/loss',
                          prefix=prefix,
                          save_dir=str(current_snapshot_dir),
                          conditions='min', model=model),

               #ItemWiseBinaryJaccardDiceMeter(prefix="eval",  name='jaccard',
                                           #   parse_output=partial(parse_binary_label, threshold=threshold),
                                            #  parse_target=lambda x: x.squeeze().to(device)),
              # ItemWiseBinaryJaccardDiceMeter(prefix="eval",  name='dice',
                                             # parse_output=partial(parse_binary_label, threshold=threshold),
                                              #parse_target=lambda x: x.squeeze().to(device)),

               # Reduce LR on plateau
               SimpleLRScheduler('eval/loss', ReduceLROnPlateau(optimizer,
                                                                patience=int(config['training']['patience']),
                                                                factor=float(config['training']['factor']),
                                                                eps=float(config['training']['eps']))),
               ScalarMeterLogger(writer=writer, comment='validation', log_dir=log_dir))

    return train_cbs, val_cbs


def init_loss(config, device='cuda'):
    # TODO new loss function: This could be MSE, L1 or Wing loss

    if config['training']['loss'] == 'bce':
        return BCEWithLogitsLoss2d().to(device)
    elif config['training']['loss'] == 'mse':
        return nn.MSELoss().to(device)
    elif config['training']['loss'] == 'jaccard':
        return SoftJaccardLoss(use_log=config['training']['log_jaccard']).to(device)
    elif config['training']['loss'] == 'combined':
        return CombinedLoss([BCEWithLogitsLoss2d(),
                            SoftJaccardLoss(use_log=config['training']['log_jaccard'])]).to(device)
    else:
        raise Exception('No compatible loss selected in experiment_config.yml! Set training->loss accordingly.')


def init_model(model_selection='ReconNet'):
    if model_selection == 'ReconNet':
        model = ReconNet()
    else:
        raise Exception('Model not implemented!')
    return model


def create_data_provider(args, config, parser, metadata, mean, std):
    # Compile ItemLoaders
    item_loaders = dict()
    for stage in ['train', 'val']:
        item_loaders[f'bfpn_{stage}'] = ItemLoader(meta_data=metadata[stage],
                                                   transform=train_test_transforms(config, mean, std)[stage],
                                                   parse_item_cb=parser,
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


def parse_item_test(root, entry, transform, data_key, target_key):
    img = cv2.imread(str(entry.fname), 0)
   # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    dc = sld.DataContainer((img, ), 'I',  transform_settings={0: {'interpolation': 'bilinear'}})
    img = transform(dc)[0]
    img = torch.cat([img, img, img], 0) / 255.
    img = img.permute(2, 0, 1) / 255.

    return {data_key: img}


def parse_color_im(root, entry, transform, data_key, target_key):
    # Image and mask generation
    img = cv2.imread(str(entry.fname))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(entry.mask_fname), 0) / 255.

    if img.shape[0] != mask.shape[0]:
        img = cv2.resize(img, (mask.shape[1], mask.shape[0]))
    elif img.shape[1] != mask.shape[1]:
        mask = mask[:, :img.shape[1]]

    img, mask = transform((img, mask))
    img = img.permute(2, 0, 1) / 255.  # img.shape[0] is the color channel after permute

    # Images are in the format 3xHxW
    # and scaled to 0-1 range
    return {data_key: img, target_key: mask}

def parse_grayscale(root, entry, transform, data_key, target_key, mean=False):
    # TODO make sure that this is working
    # Image and mask generation
    img = cv2.imread(str(entry.fname))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img[:, :, 1] = img[:, :, 0]
    img[:, :, 2] = img[:, :, 0]
    #plt.imshow(img[:,:,0])
    #plt.show()

    mask = cv2.imread(str(entry.mask_fname))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask[:, :, 1] = mask[:, :, 0]
    mask[:, :, 2] = mask[:, :, 0]

    #plt.imshow(mask[1, :, :])
    #plt.show()

    if img.shape[0] != mask.shape[0]:
        img = cv2.resize(img, (mask.shape[1], mask.shape[0]))
    elif img.shape[1] != mask.shape[1]:
        mask = mask[:, :img.shape[1]]

    img, mask = transform((img, mask))
    img = img[:, :, 0]
    img = np.expand_dims(img, 1)
    img = np.transpose(img, (1,2, 0))

    mask = mask[:, :, 0] / 90.
    mask = np.expand_dims(mask, 1)
    mask = np.transpose(mask, (1,2, 0))
    return {data_key: img, target_key: mask}

    #img = img.permute(2, 0, 1)# / 255.  # img.shape[0] is the color channel after permute
    #mask = mask.permute(2, 0, 1)
    #mask = ndimage.zoom(mask, 1.0)
    #img = ndimage.zoom(img, 1.0)

   # plt.imshow(mask[0,:])
    #plt.show()

    #plt.imshow(img[0,:])
    #plt.show()


    # Debugging
    #plt.imshow(np.asarray(img).transpose((1, 2, 0)))
    #plt.imshow(np.asarray(mask).squeeze(), alpha=0.3)
    #plt.show()
    
    # Calculate mean profile
'''
    if mean:
        mean_mask = np.mean(mask.numpy(), axis=2)
        mean_mask = np.expand_dims(mean_mask, 1)
        mean_mask = ndimage.zoom(mean_mask, 1.0)
        mean_mask = torch.Tensor(mean_mask)
        mask = mean_mask


        mean_img = np.mean(img.numpy(), axis=2)
        mean_img = np.expand_dims(mean_img, 1)
        mean_img = ndimage.zoom(mean_img, 1.0)
        mean_img = torch.Tensor(mean_img)
        img = mean_img

        #plt.plot(mean_img[0,:])
        #plt.show()

    # Images are in the format 3xHxW
    # and scaled to 0-1 range
    return {data_key: img, target_key: mask}

def parse_grayscale(root, entry, transform, data_key, target_key):
    # TODO make sure that this is working
    # Image and mask generation

    img = cv2.imread(str(entry.fname))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img[:, :, 1] = img[:, :, 0]
    img[:, :, 2] = img[:, :, 0]
    # mask = cv2.imread(str(entry.mask_fname), 0) / 255.
    mask = cv2.imread(str(entry.mask_fname))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
   # plt.imshow(mask)
    #plt.show()
    mask[:, :, 1] = mask[:, :, 0]
    mask[:, :, 2] = mask[:, :, 0]

    if img.shape[0] != mask.shape[0]:
        img = cv2.resize(img, (mask.shape[1], mask.shape[0]))
    elif img.shape[1] != mask.shape[1]:
        mask = mask[:, :img.shape[1]]

    img, mask = transform((img, mask))
   # img = img.permute(2, 0, 1) / 255.  # img.shape[0] is the color channel after permute
   # mask = mask.permute(2, 0, 1)

    # Debugging
    #plt.imshow(np.asarray(img).transpose((1, 2, 0)))
    #plt.imshow(np.asarray(mask).squeeze(), alpha=0.3)
    #plt.show()

    # Images are in the format 3xHxW
    # and scaled to 0-1 range
    return {data_key: img, target_key: mask}

'''
def save_config(path, config, args, model):
    """
    Alternate way to save model parameters.
    """
    with open(path / 'experiment_config.txt', 'w') as f:
        f.write(f'\nArguments file:\n')
        f.write(f'Seed: {args.seed}\n')
        f.write(f'Batch size: {args.bs}\n')
        f.write(f'N_epochs: {args.n_epochs}\n')

        FILE='model.pth'
        torch.save(model.state_dict(), FILE)
        f.write('Configuration file:\n\n')
        for key, val in config.items():
            f.write(f'{key}\n')
        #    for key2 in config[key].items():
         #       f.write(f'\t{key2}\n')


def save_transforms(path, config, args, mean, std):
    transforms = train_test_transforms(config, mean, std)
    # Save the experiment parameters
    with open(path / 'transforms.json', 'w') as f:
        f.writelines(json.dumps(transforms['train_list'][1].serialize(), indent=4))

