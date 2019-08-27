import pandas as pd
import numpy as np
import pathlib
import cv2
import torch
from sklearn import model_selection

from rabbitccs.data.transforms import estimate_mean_std


def build_meta_from_files(base_path):
    masks_loc = base_path / 'masks'
    images_loc = base_path / 'images'

    images = set(map(lambda x: x.stem, images_loc.glob('*.png')))
    masks_train = set(map(lambda x: x.stem, masks_loc.glob('*.png')))  # Path was modified to include all samples
    masks_test = set(map(lambda x: x.stem, masks_loc.glob('masks_test/*.png')))
    masks = masks_train.union(masks_test)

    res = masks.intersection(images)
    assert len(res), len(masks)

    masks_train = list(map(lambda x: pathlib.Path(x).with_suffix('.png'), masks_train))
    masks_test = list(map(lambda x: pathlib.Path(x).with_suffix('.png'), masks_test))

    d_train = {'fname': [], 'mask_fname': []}
    d_test = {'fname': [], 'mask_fname': []}

    # Making train dataframe
    [d_train['fname'].append((images_loc / img_name)) for img_name in masks_train]
    [d_train['mask_fname'].append(masks_loc / img_name) for img_name in masks_train]

    # Making test dataframe
    [d_test['fname'].append(images_loc / img_name) for img_name in masks_test]
    [d_test['mask_fname'].append(masks_loc / 'masks_test' / img_name) for img_name in masks_test]

    train = pd.DataFrame(data=d_train)
    test = pd.DataFrame(data=d_test)

    return train, test


def build_splits(data_dir, args, config, snapshots_dir):
    # Metadata
    train, test = build_meta_from_files(data_dir)
    train['subj_id'] = train.fname.apply(lambda x: x.stem.split('_')[0] + '_' + x.stem.split('_')[1], 0)  # Group_ID

    # Mean and std
    crop = config['training']['crop_size']
    mean_std_path = snapshots_dir / f"mean_std_{crop[0]}x{crop[1]}.pth"
    if mean_std_path.is_file():
        print('==> Loading mean and std from cache')
        tmp = torch.load(mean_std_path)
        mean, std = tmp['mean'], tmp['std']
    else:
        print('==> Estimating mean and std')
        mean, std = estimate_mean_std(config, train, parse_item_cb, args.num_threads, args.bs)
        torch.save({'mean': mean, 'std': std}, mean_std_path)

    print('==> Mean:', mean)
    print('==> STD:', std)

    # Group K-Fold by rabbit ID
    gkf = model_selection.GroupKFold(n_splits=config['training']['n_folds'])

    # Create splits for all folds
    splits_list = []
    iterator = gkf.split(train, groups=train.subj_id)
    for i in range(config['training']['n_folds']):
        train_idx, val_idx = next(iterator)
        metadata = {'train': train.iloc[train_idx],  # Swapped train and val idx
                    'val': train.iloc[val_idx],
                    'test': test}
        splits_list.append(metadata)

    return splits_list, mean, std


def parse_item_cb(root, entry, transform, data_key, target_key):
    # Image and mask generation
    img = cv2.imread(str(entry.fname))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = np.transpose(img, (2, 0, 1))
    mask = cv2.imread(str(entry.mask_fname), 0) / 255.

    if img.shape[0] != mask.shape[0]:
        img = cv2.resize(img, (mask.shape[1], mask.shape[0]))
    img, mask = transform((img, mask))
    # img = torch.cat([img, img, img], 0)
    # mask = torch.cat(mask.unsqueeze(0))
    img = img.permute(2, 0, 1) / 255.  # img.shape[0] is the color channel after permute

    # Images are in the format 3xHxW
    # and scaled to 0-1 range
    return {data_key: img, target_key: mask}
