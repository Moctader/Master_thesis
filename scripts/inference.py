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
    parser.add_argument('--input_x', type=int, default=1024)
    parser.add_argument('--input_y', type=int, default=512)
    args = parser.parse_args()

    # Load configuration file
    with open(args.experiment, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Load models
    models = glob(str(args.snapshots_root) + '/fold_*')
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
    input_x = args.input_x
    input_y = args.input_y
    for file in tqdm(files, desc='Running inference'):
        img_full = cv2.imread(file)
        x, y, ch = img_full.shape
        mask_full = np.zeros((x, y))

        # Check for amount of blocks
        # residual_x = int((x % 1024 > 0))
        residual_y = int((y % input_y > 0))
        blocks_y = img_full.shape[1] // input_y + residual_y


        # Loop evaluating through large image
        for i in range(blocks_y):
            # Select part of the large image for network
            if img_full.shape[0] >= input_x:
                if blocks_y == 1:  # x >= 1024, y < 512
                    im_pad = np.zeros((input_x, input_y, 3))
                    im_pad[:, :y, :] = img_full[:input_x, :, :]
                    img = im_pad[:, :, :]
                else:  # x >= 1024, y >= 512
                    if (blocks_y == i + 1) & bool(residual_y):  # end of the large image
                        img = img_full[:input_x, - input_y:, :]
                    else:
                        img = img_full[:input_x, i * input_y:(i + 1) * input_y, :]
            else:
                im_pad = np.zeros((input_x, input_y, 3))
                if blocks_y == 1:  # x < 1024, y < 512
                    im_pad[:x, :y, :] = img_full[:, :, :]

                else:  # x < 1024, y >= 512
                    if (blocks_y == i + 1) & bool(residual_y):  # end of the large image
                        im_pad[:x, :, :] = img_full[:, - input_y:, :]
                    else:
                        im_pad[:x, :, :] = img_full[:, i * input_y:(i + 1) * input_y, :]
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
            if x >= input_x:
                if (blocks_y == i + 1) & bool(residual_y):  # end of the large image
                    mask_full[:input_x, - (y % input_y):] = mask_final[:, - (y % input_y):]
                else:
                    mask_full[:input_x, i * input_y:(i + 1) * input_y] = mask_final
            else:
                if (blocks_y == i + 1) & bool(residual_y):  # end of the large image
                    mask_full[:, - (y % input_y):] = mask_final[:input_x, - (y % input_y):]
                else:
                    mask_full[:, i * input_y:(i + 1) * input_y] = mask_final[:input_x, :]



        cv2.imwrite(str(args.save_dir) + '/' + file.split('/')[-1], mask_full)
