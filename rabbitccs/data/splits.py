import pandas as pd
import numpy as np
import pathlib
import cv2
import torch
from sklearn import model_selection


def build_meta_from_files(base_path):
    # TODO: this function (Done)
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


def build_splits(data_dir):
    # TODO: this function (Done)
    train, test = build_meta_from_files(data_dir)

    train['subj_id'] = train.fname.apply(lambda x: x.stem.split('_')[0] + '_' + x.stem.split('_')[1], 0)  # Group_ID
    # TODO: group K-Fold by rabbit ID (Done)
    gkf = model_selection.GroupKFold(n_splits=12)  # Splits equal to n_rabbits
    train_idx, val_idx = next(gkf.split(train, groups=train.subj_id))

    return {'train': train.iloc[train_idx],  # Swapped train and val idx
            'val': train.iloc[val_idx],
            'test': test}


def parse_item_cb(root, entry, transform, data_key, target_key):
    # TODO: this function (mask generation) (Done)
    img = cv2.imread(str(entry.fname))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = np.transpose(img, (2, 0, 1))
    mask = cv2.imread(str(entry.mask_fname), 0) / 255.

    #TODO: Check and mask with the debugger
    if img.shape[0] != mask.shape[0]:
        img = cv2.resize(img, (mask.shape[1], mask.shape[0]))
    img, mask = transform((img, mask))
    # img = torch.cat([img, img, img], 0)
    # mask = torch.cat(mask.unsqueeze(0))
    img = img.permute(2, 0, 1) / 255. # img.shape[0] is the color channel after permute

    # TODO: Chek that the images are in the format 3xHxW (Done)
    # TODO: img must be scaled (Done)
    return {data_key: img, target_key: mask}
