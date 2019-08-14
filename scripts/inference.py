import cv2
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import argparse
import os
import torch
import yaml
from tqdm import tqdm
from glob import glob
from collagen.modelzoo.segmentation import EncoderDecoder
from collagen.core.utils import auto_detect_device
from rabbitccs.data.transforms import numpy2tens

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset_root', default='../../../Data/images/')
    parser.add_argument('--dataset_root', type=pathlib.Path, default='/media/dios/dios2/HistologySegmentation/Images/')
    parser.add_argument('--save_dir', type=pathlib.Path, default='/media/dios/dios2/HistologySegmentation/Predictions/')
    parser.add_argument('--tta', type=bool, default=False)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--experiment', default='./experiment_config.yml')
    parser.add_argument('--n_threads', type=int, default=12)
    parser.add_argument('--snapshots_root', default='../../../workdir/snapshots/')
    parser.add_argument('--snapshot', default='')
    args = parser.parse_args()

    # Load configuration file
    with open(args.experiment, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Load model
    device = auto_detect_device()
    model = EncoderDecoder(**config['model']).to(device)
    model.load_state_dict(torch.load(os.path.join(args.snapshots_root, 'model_0018_20190813_170835_eval.loss_0.093.pth')))
    model.eval()

    # Find image files
    files = glob(str(args.dataset_root) + '/*.png')
    args.save_dir.mkdir(exist_ok=True)

    # Loop for all images
    for file in tqdm(files, desc='Running inference'):
        img_full = cv2.imread(file)
        x, y, z = img_full.shape
        mask_full = np.zeros(img_full.shape)
        # for i in range(img_full.shape[0] % 1024):
        if img_full.shape[0] >= 1024:
            img = img_full[:1024, :512, :]
        else:
            im_pad = np.zeros((1024, 512, 3))
            im_pad[:img_full.shape[0], :, :] = img_full[:, :512, :]
            img = im_pad[:, :, :]

        img = numpy2tens(img / 255.).permute(2, 0, 1).to('cuda')
        mask = torch.sigmoid(model(img.unsqueeze(0))).mul(255).to('cpu').detach().numpy().astype('uint8').squeeze()
        mask = (mask >= 255 // 2).astype('uint8') * 255
        cv2.imwrite(str(args.save_dir) + '/' + file.split('/')[-1], mask)
