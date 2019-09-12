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
    parser.add_argument('--dataset_root', type=pathlib.Path, default='../../../Data/images_test/')
    parser.add_argument('--save_dir', type=pathlib.Path, default='../../../Data/predictions_test/')
    parser.add_argument('--tta', type=bool, default=False)
    parser.add_argument('--experiment', default='./experiment_config.yml')
    parser.add_argument('--snapshots_root', default='../../../workdir/snapshots/')
    parser.add_argument('--snapshot', default='dios-erc-gpu_2019_08_27_17_59_41/')
    parser.add_argument('--crop_size', type=tuple, default=(512, 1024))
    args = parser.parse_args()

    # Load configuration file
    with open(args.experiment, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Load models
    models = glob(str(args.snapshots_root) + str(args.snapshot) + '/*fold_*.pth')
    models.sort()
    device = auto_detect_device()

    # List the models
    model_list = []
    for fold in range(len(models)):
        model = EncoderDecoder(**config['model']).to(device)
        model.load_state_dict(torch.load(models[fold]))
        model.eval()
        model_list.append(model)

    # Find image files
    files = glob(str(args.dataset_root) + '/*.png')
    args.save_dir.mkdir(exist_ok=True)

    # Loop for all images
    crop = config['training']['crop_size']
    for file in tqdm(files, desc='Running inference'):
        img_full = cv2.imread(file)
        x, y, ch = img_full.shape
        mask_full = np.zeros((x, y))

        # Check for amount of blocks
        residuals = (int(x % crop[0] > 0), int(y % crop[1] > 0))
        blocks_x = img_full.shape[0] // crop[0] + residuals[0]
        blocks_y = img_full.shape[1] // crop[1] + residuals[1]

        # Loop evaluating through large image
        for i in range(blocks_y):
            # Select part of the large image for network
            if img_full.shape[0] >= crop[0]:
                if blocks_y == 1:  # x >= 1024, y < 512
                    im_pad = np.zeros((crop[0], crop[1], 3))
                    im_pad[:, :y, :] = img_full[:crop[0], :, :]
                    img = im_pad[:, :, :]
                else:  # x >= 1024, y >= 512
                    if (blocks_y == i + 1) & bool(residuals[1]):  # end of the large image
                        img = img_full[:crop[0], - crop[1]:, :]
                    else:
                        img = img_full[:crop[0], i * crop[1]:(i + 1) * crop[1], :]
            else:
                im_pad = np.zeros((crop[0], crop[1], 3))
                if blocks_y == 1:  # x < 1024, y < 512
                    im_pad[:x, :y, :] = img_full[:, :, :]

                else:  # x < 1024, y >= 512
                    if (blocks_y == i + 1) & bool(residuals[1]):  # end of the large image
                        im_pad[:x, :, :] = img_full[:, - crop[1]:, :]
                    else:
                        im_pad[:x, :, :] = img_full[:, i * crop[1]:(i + 1) * crop[1], :]
                img = im_pad[:, :, :]
            # Convert image to tensor
            img = numpy2tens(img / 255.).permute(2, 0, 1).to('cuda')

            # Loop evaluating inference on every fold
            masks = []
            for fold in range(len(models)):
                # Inference
                mask = torch.sigmoid(model_list[fold](img.unsqueeze(0))).to('cpu').detach().numpy().astype('float32').squeeze()
                masks.append(mask)

            # Average of predictions
            mask_mean = np.mean(masks, 0)
            mask_final = (mask_mean >= 0.5).astype('uint8') * 255

            # Add prediction to large mask
            if x >= crop[0]:
                if (blocks_y == i + 1) & bool(residuals[1]):  # end of the large image
                    mask_full[:crop[0], - (y % crop[1]):] = mask_final[:, - (y % crop[1]):]
                else:
                    mask_full[:crop[0], i * crop[1]:(i + 1) * crop[1]] = mask_final
            else:
                if (blocks_y == i + 1) & bool(residuals[1]):  # end of the large image
                    mask_full[:, - (y % crop[1]):] = mask_final[:crop[0], - (y % crop[1]):]
                else:
                    mask_full[:, i * crop[1]:(i + 1) * crop[1]] = mask_final[:crop[0], :]

        # Save predicted full mask
        cv2.imwrite(str(args.save_dir) + '/' + file.split('/')[-1], mask_full)
