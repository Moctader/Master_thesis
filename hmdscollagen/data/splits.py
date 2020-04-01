from lib2to3.pytree import convert
import numpy as np
import cv2
import pandas as pd
import pathlib
import torch
import dill
from sklearn import model_selection
from hmdscollagen.data.transforms import estimate_mean_std
import glob
import os
from glob import glob

def build_meta_from_files(base_path, phase='train'):
    # Path to images
    if phase == 'train':
        masks_loc = pathlib.Path(base_path) / 'plm'
        images_loc = pathlib.Path(base_path) / 'hmds'
        # images_loc = base_path / 'images'
    else:
        masks_loc = base_path / 'predictions_test'
        images_loc = base_path / 'images_test'


        # List files
    images = set(map(lambda x: x.stem, images_loc.glob('**/*[0-9].[pb][nm][gp]')))  # HMDS images
    #masks = set(map(lambda x: x.stem, masks_loc.glob('*a.mat.png')))
    #masks = set(map(lambda x: x.stem, masks_loc.glob('*.png')))
    # l = [masks]*300
    #masks_a = [set(map(lambda x: x.stem, masks_loc.glob('*a.mat.png')))]
    #masks_b = [set(map(lambda x: x.stem, masks_loc.glob('*b.mat.png')))]
    #masks = (masks_a + masks_b)
    #masks = tuple(masks_list)

    # masks_f = set((masks))



    # res = masks.intersection(images)

    # masks = list(map(lambda x: pathlib.Path(x).with_suffix('.png'), masks))
    images = list(map(lambda x: pathlib.Path(x.name), images_loc.glob('**/*[0-9].[pb][nm][gp]')))
    masks = list(map(lambda x: pathlib.Path(x.name), masks_loc.glob('*.png')))
    images.sort()
    masks.sort()

    # Repeat the PLM images 300 times
    masks_repeat = []
    for i in range(len(masks)):
        hmds_images = glob(str(images_loc) + '/*' + str(masks[i])[:-9] + '/*.png', recursive=True)
        if '6061-12La.mat.png' in str(masks[i]):
            masks_repeat.extend(list(np.repeat(masks[i], len(hmds_images))))
        elif 'a.mat.png' in str(masks[i]):
            masks_repeat.extend(list(np.repeat(masks[i], len(hmds_images) // 2)))
        elif 'b.mat.png' in str(masks[i]):
            masks_repeat.extend(list(np.repeat(masks[i], len(hmds_images) // 2 + len(hmds_images) % 2)))
        else:
            raise Exception('Wrong file name for the PLM image')

    #assert len(res), len(masks)
    d_frame = {'fname': [], 'mask_fname': []}
    # Making dataframe
    #[d_frame['fname'].append((images_loc / str(img_name)[:7].rsplit('_', 1)[0] / img_name)) for img_name in images]
    [d_frame['fname'].append((images_loc / str(img_name)[:8].rsplit('_', 1)[0] / img_name)) for img_name in images]
    [d_frame['mask_fname'].append(masks_loc / img_name) for img_name in masks_repeat]

    metadata = pd.DataFrame(data=d_frame)

    return metadata


def build_splits(data_dir, args, config, parser, snapshots_dir, snapshot_name):
    # TODO correct splits
    # Metadata
    # metadata = build_meta_from_files(data_dir)
    metadata = build_meta_from_files(base_path=data_dir)
    # Group_ID
    if config['training']['uCT']:
        metadata['subj_id'] = metadata.fname.apply(lambda x: x.stem.rsplit('_', 2)[0], 0)
    else:
        metadata['subj_id'] = metadata.fname.apply(lambda x: x.stem.rsplit('_', 3)[0], 0)

    # Mean and std
    crop = config['training']['crop_size']
    mean_std_path = snapshots_dir / f"mean_std_{crop[0]}x{crop[1]}.pth"
    if mean_std_path.is_file() and not config['training']['calc_meanstd']:  # Load
        print('==> Loading mean and std from cache')
        tmp = torch.load(mean_std_path)
        mean, std = tmp['mean'], tmp['std']
    else:  # Calculate
        print('==> Estimating mean and std')
        mean, std = estimate_mean_std(config, metadata, parser, args.num_threads, args.bs)
        torch.save({'mean': mean, 'std': std}, mean_std_path)

    print('==> Mean:', mean)
    print('==> STD:', std)

    # Group K-Fold by rabbit ID
    gkf = model_selection.GroupKFold(n_splits=config['training']['n_folds'])
    # K-fold by random shuffle
    # gkf = model_selection.KFold(n_splits=config['training']['n_folds'], shuffle=True, random_state=args.seed)

    # Create splits for all folds
    splits_metadata = dict()
    iterator = gkf.split(metadata, groups=metadata.subj_id)
    for fold in range(config['training']['n_folds']):
        train_idx, val_idx = next(iterator)
        splits_metadata[f'fold_{fold}'] = {'train': metadata.iloc[train_idx],
                                           'val': metadata.iloc[val_idx]}

    # Add mean and std to metadata
    splits_metadata['mean'] = mean
    splits_metadata['std'] = std

    with open(snapshots_dir / snapshot_name / 'split_config.dill', 'wb') as f:
        dill.dump(splits_metadata, f)

    return splits_metadata
