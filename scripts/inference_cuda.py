import argparse
import torch
from pathlib import Path
import dill
import yaml
import numpy as np
from tqdm import tqdm
import cv2
import pandas as pd

from collagen.data import ItemLoader
from collagen.modelzoo.segmentation import EncoderDecoder

from rabbitccs.data.transforms import train_test_transforms
from rabbitccs.training.session import parse_item_test, parse_binary_label
from rabbitccs.data.utilities import mask2rle
from rabbitccs.data.splits import build_meta_from_files

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_location', type=Path, default='../../../Data/')
    parser.add_argument('--snapshot', type=Path, default='../../../workdir/snapshots/dios-erc-gpu_2019_08_30_13_42_14/')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_threads', type=int, default=16)
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--sum_threshold', type=int, default=15)
    parser.add_argument('--multi_gpu', type=bool, default=False)
    args = parser.parse_args()

    with open(args.snapshot / 'config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    with open(args.snapshot / 'args.dill', 'rb') as f:
        args_experiment = dill.load(f)

    with open(args.snapshot / 'split_config.dill', 'rb') as f:
        split_config = dill.load(f)

    mean, std = split_config['mean'], split_config['std']
    metadata = build_meta_from_files(args.data_location, phase='test')
    test_loader = ItemLoader(meta_data=metadata,
                             transform=train_test_transforms(config, mean, std)['test'],
                             # parse_item_cb=parse_binary_label,
                             parse_item_cb=parse_item_test,
                             batch_size=args.bs,
                             num_workers=args.num_threads, shuffle=False)

    threshold = 0.5 if config['training']['log_jaccard'] is False else 0.3

    for fold_id in range(config['training']['n_folds']):
        snp = list(args.snapshot.glob(f"*fold_{fold_id}*.pth"))
        if len(snp) == 0:
            continue
        else:
            snp = snp[0]

        (args.snapshot / 'test_inference' / f'fold_{fold_id}').mkdir(parents=True, exist_ok=True)

        model = EncoderDecoder(**config['model'])
        if args.multi_gpu:
            model = torch.nn.DataParallel(model)

        model.load_state_dict(torch.load(snp))
        print(f'==> Loaded the snapshot {snp}')
        if not args.multi_gpu and isinstance(model, torch.nn.DataParallel):
            model = model.module

        model.to('cuda')
        model.eval()
        #metadata_idx = metadata.set_index('ImageID')
        #submission = {'ImageId': [], 'EncodedPixels': []}
        with torch.no_grad():
            for batch_idx in tqdm(range(len(test_loader))):
                batch = test_loader.sample()[0]
                out = torch.sigmoid(model(batch['data'].to('cuda')))
                out_cpu = out.to('cpu').numpy()
                img_ids = batch['ImageId']

                for i in range(out_cpu.shape[0]):
                    entry = metadata.loc[img_ids[i]]
                    h, w = entry.h, entry.w
                    mask_i = out_cpu[i].squeeze()
                    save_pic = args.snapshot / 'test_inference' / f'fold_{fold_id}' / (img_ids[i]+'.png')
                    cv2.imwrite(str(save_pic), np.uint8(mask_i*255))
                    mask_i_t = mask_i > threshold
                    decoded = None
                    if mask_i_t.sum() < args.sum_threshold:
                        decoded = "-1"
                        mask_resized = np.zeros((h, w), dtype=np.uint8)
                    else:
                        if mask_i_t.shape != (h, w):
                            mask_resized = cv2.resize(np.uint8(mask_i_t)*255,
                                                      (w, h),
                                                      interpolation=cv2.INTER_NEAREST)
                        else:
                            mask_resized = np.uint8(mask_i_t)*255
                        decoded = mask2rle(mask_resized, w, h)
                    submission['ImageId'].append(img_ids[i])
                    submission['EncodedPixels'].append(decoded)

        submission = pd.DataFrame(data=submission)
        save_fold_submission = args.snapshot / 'test_inference' / f'{args.snapshot.stem}_fold_{fold_id}.csv'
        submission.to_csv(save_fold_submission, index=None)





