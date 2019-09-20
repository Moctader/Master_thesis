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
from time import sleep, time
from tqdm import tqdm
from glob import glob
from collagen.modelzoo.segmentation import EncoderDecoder
from collagen.core.utils import auto_detect_device

from rabbitccs.data.utilities import load, save
from rabbitccs.data.visualizations import render_volume

from pytorch_toolbelt.inference.tiles import ImageSlicer, CudaTileMerger
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image, to_numpy

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def inference(img_full, device=1):
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
            pred_batch = torch.sigmoid(model_list[fold](tiles_batch)).detach()

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

    # Free memory
    torch.cuda.empty_cache()
    gc.collect()

    return mask_mean.squeeze()


if __name__ == "__main__":
    start = time()

    parser = argparse.ArgumentParser()
    #parser.add_argument('--dataset_root', type=Path, default='/media/dios/dios2/RabbitSegmentation/µCT/images')
    #parser.add_argument('--save_dir', type=Path, default='/media/dios/dios2/RabbitSegmentation/µCT/predictions_5_fold/')
    parser.add_argument('--dataset_root', type=Path, default='../../../Data/µCT/images')
    parser.add_argument('--save_dir', type=Path, default='/media/dios/dios2/RabbitSegmentation/µCT/predictions_5_fold_trainingset/')
    parser.add_argument('--bs', type=int, default=5)
    parser.add_argument('--plot', type=bool, default=False)
    parser.add_argument('--weight', type=str, choices=['pyramid', 'mean'], default='mean')
    parser.add_argument('--experiment', default='./experiment_config_uCT.yml')
    parser.add_argument('--snapshot', type=Path,
                        default='../../../workdir/snapshots/dios-erc-gpu_2019_09_18_15_32_33_8samples/')
    parser.add_argument('--dtype', type=str, choices=['.bmp', '.png', '.tif'], default='.bmp')
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
    #models = glob(str(args.snapshot) + '/*fold_3_*.pth')
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
    print(f'Found {len(model_list)} models.')

    # Loop for samples
    args.save_dir.mkdir(exist_ok=True)
    samples = os.listdir(args.dataset_root)
    for sample in samples:
        try:
            sleep(0.5); print(f'==> Processing sample: {sample}')

            # Load image stacks
            data_xz, files = load(str(args.dataset_root / sample), rgb=True)
            data_xz = np.transpose(data_xz, (1, 0, 2, 3))  # X-Z-Y-Ch
            data_yz = np.transpose(data_xz, (0, 2, 1, 3))  # Y-Z-X-Ch
            mask_xz = np.zeros(data_xz.shape)[:, :, :, 0]  # Remove channel dimension
            mask_yz = np.zeros(data_yz.shape)[:, :, :, 0]

            threshold = 0.5 if config['training']['log_jaccard'] is False else 0.3

            # Loop for image slices
            input_x = config['training']['crop_size'][0]
            input_y = config['training']['crop_size'][1]
            # 1st orientation
            for slice in tqdm(range(data_xz.shape[2]), desc='Running inference, XZ'):
                mask_xz[:, :, slice] = inference(data_xz[:, :, slice, :])
            # 2nd orientation
            for slice in tqdm(range(data_yz.shape[2]), desc='Running inference, YZ'):
                mask_yz[:, :, slice] = inference(data_yz[:, :, slice, :])

            # Average probability maps
            mask_final = ((mask_xz + np.transpose(mask_yz, (0, 2, 1))) / 2) >= threshold

            # Convert to original orientation
            mask_final = np.transpose(mask_final, (0, 2, 1)).astype('uint8') * 255

            # Save predicted full mask
            save(str(args.save_dir / sample), files, mask_final, dtype=args.dtype)

            render_volume(data_yz[:, :, :, 0] * mask_final,
                          savepath=str(args.save_dir / 'visualizations' / (sample + '.png')),
                          white=True, use_outline=False)
        except:
            print(f'Sample {sample} failed.')
            continue

    print(f'Inference completed in {(time() - start) // 60} minutes, {(time() - start) % 60} seconds.')
