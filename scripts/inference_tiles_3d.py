import cv2
import numpy as np
import gc
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import dill
import torch
import yaml
from time import sleep
from tqdm import tqdm
from glob import glob
from collagen.modelzoo.segmentation import EncoderDecoder
from collagen.core.utils import auto_detect_device
from rabbitccs.data.transforms import numpy2tens

from pytorch_toolbelt.inference.tiles import ImageSlicer, CudaTileMerger
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image, to_numpy

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=Path, default='/media/dios/dios2/RabbitSegmentation/µCT/images')
    parser.add_argument('--save_dir', type=Path, default='/media/dios/dios2/RabbitSegmentation/µCT/predictions/')
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--plot', type=bool, default=False)
    parser.add_argument('--weight', type=str, choices=['pyramid', 'mean'], default='mean')
    parser.add_argument('--experiment', default='./experiment_config_uCT.yml')
    parser.add_argument('--snapshot', type=Path, default='../../../workdir/snapshots/dios-erc-gpu_2019_09_12_14_34_22/')
    args = parser.parse_args()

    # Load snapshot configuration
    with open(args.snapshot / 'config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    with open(args.snapshot / 'args.dill', 'rb') as f:
        args_experiment = dill.load(f)

    with open(args.snapshot / 'split_config.dill', 'rb') as f:
        split_config = dill.load(f)

    # Load models
    models = glob(str(args.snapshot) + '/*fold_[0-9]_*.pth')
    models.sort()
    #device = auto_detect_device()
    device = 1  # Use the second GPU for inference

    # List the models
    model_list = []
    for fold in range(len(models)):
        model = EncoderDecoder(**config['model']).to(device)
        model.load_state_dict(torch.load(models[fold]))
        model.eval()
        model_list.append(model)

    # Loop for samples
    args.save_dir.mkdir(exist_ok=True)
    samples = os.listdir(args.dataset_root)
    for sample in samples:
        sleep(0.5); print(f'==> Processing sample: {sample}')
        # Find image files
        files = glob(str(args.dataset_root / sample / '*[0-9].[pb][nm][gp]'))

        threshold = 0.5 if config['training']['log_jaccard'] is False else 0.3

        # Loop for image slices
        input_x = config['training']['crop_size'][0]
        input_y = config['training']['crop_size'][1]
        for file in tqdm(files, desc='Running inference on slices'):

            img_full = cv2.imread(file)
            x, y, ch = img_full.shape
            mask_full = np.zeros((x, y))

            # Cut large image into overlapping tiles
            tiler = ImageSlicer(img_full.shape, tile_size=(input_x, input_y),
                                tile_step=(input_x // 2, input_y // 2), weight=args.weight)

            # HCW -> CHW. Optionally, do normalization here
            tiles = [tensor_from_rgb_image(tile) for tile in tiler.split(img_full)]

            # Allocate a CUDA buffer for holding entire mask
            merger = CudaTileMerger(tiler.target_shape, channels=1, weight=tiler.weight, device=device)

            # Loop evaluating inference on every fold
            masks = []
            for fold in range(len(models)):

                # Run predictions for tiles and accumulate them
                for tiles_batch, coords_batch in DataLoader(list(zip(tiles, tiler.crops)), batch_size=args.bs, pin_memory=True):
                    # Move tile to GPU
                    tiles_batch = (tiles_batch.float() / 255.).to(device)
                    # Predict and move back to CPU
                    pred_batch = torch.sigmoid(model_list[fold](tiles_batch))  # .detach()

                    # Merge on GPU
                    merger.integrate_batch(pred_batch, coords_batch)

                    # Plot
                    if args.plot:
                        for i in range(args.bs):
                            if args.bs != 1:
                                plt.imshow(pred_batch.cpu().detach().numpy().astype('float32').squeeze()[i, :, :])
                            else:
                                plt.imshow(pred_batch.cpu().detach().numpy().astype('float32').squeeze())
                            plt.show()

                # Normalize accumulated mask and convert back to numpy
                merged_mask = np.moveaxis(to_numpy(merger.merge()), 0, -1).astype('float32')
                merged_mask = tiler.crop_to_orignal_size(merged_mask)
                # Plot
                if args.plot:
                    for i in range(args.bs):
                        if args.bs != 1:
                            plt.imshow(merged_mask)
                        else:
                            plt.imshow(merged_mask.squeeze())
                        plt.show()
                masks.append(merged_mask)

            # Average of predictions
            mask_mean = np.mean(masks, 0)
            mask_final = (mask_mean >= threshold).astype('uint8') * 255

            # Save predicted full mask
            (args.save_dir / sample).mkdir(exist_ok=True)
            cv2.imwrite(str(args.save_dir / sample / file.split('/')[-1][:-4]) + '.bmp', mask_final)
            #  cv2.imwrite(str(args.save_dir / sample / file.split('/')[-1]), mask_final)
            # Free memory
            torch.cuda.empty_cache()
            gc.collect()