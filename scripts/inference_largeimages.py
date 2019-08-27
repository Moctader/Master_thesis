import cv2
import numpy as np
import gc
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
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

from pytorch_toolbelt.inference.tiles import ImageSlicer, CudaTileMerger
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image, to_numpy

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=pathlib.Path, default='../../../Data/images_test/')
    parser.add_argument('--save_dir', type=pathlib.Path, default='../../../Data/predictions_test/')
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--experiment', default='./experiment_config.yml')
    parser.add_argument('--snapshots_root', default='../../../workdir/snapshots/5-fold_addedtransforms_512x1024/')
    parser.add_argument('--tile_size', type=tuple, default=(512, 512))
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
    input_x = args.tile_size[0]
    input_y = args.tile_size[1]
    for file in tqdm(files, desc='Running inference'):
        # Free memory
        torch.cuda.empty_cache()
        gc.collect()

        img_full = cv2.imread(file)
        x, y, ch = img_full.shape
        mask_full = np.zeros((x, y))

        # Cut large image into overlapping tiles
        tiler = ImageSlicer(img_full.shape, tile_size=(input_x, input_y),
                            tile_step=(input_x // 2, input_y // 2), weight='pyramid')

        # HCW -> CHW. Optionally, do normalization here
        tiles = [tensor_from_rgb_image(tile) for tile in tiler.split(img_full)]

        # Allocate a CUDA buffer for holding entire mask
        merger = CudaTileMerger(tiler.target_shape, 1, tiler.weight)

        # Loop evaluating inference on every fold
        masks = []
        for fold in range(len(models)):
            # Run predictions for tiles and accumulate them
            preds = []
            for tiles_batch, coords_batch in DataLoader(list(zip(tiles, tiler.crops)), batch_size=args.bs, pin_memory=True):
                # Move tile to GPU
                tiles_batch = (tiles_batch.float() / 255.).cuda()
                # Predict and move back to CPU
                pred_batch = torch.sigmoid(model_list[fold](tiles_batch))
                # preds.append(pred_batch.cpu().detach().numpy().astype('float32')[:, :, :, 0])
                # Merge on GPU
                merger.integrate_batch(pred_batch, coords_batch)

                # Plot
                for i in range(args.bs):
                    if args.bs != 1:
                        plt.imshow(pred_batch.cpu().detach().numpy().astype('float32').squeeze()[i, :, :])
                    else:
                        plt.imshow(pred_batch.cpu().detach().numpy().astype('float32').squeeze())
                    plt.show()

            # Combine to large image (CPU)
            # merged_mask = tiler.merge(preds)

            # Normalize accumulated mask and convert back to numpy
            merged_mask = np.moveaxis(to_numpy(merger.merge()), 0, -1).astype(np.uint8)
            merged_mask = tiler.crop_to_orignal_size(merged_mask)
            masks.append(merged_mask)

        # Average of predictions
        mask_mean = np.mean(masks, 0)
        mask_final = (mask_mean >= 0.5).astype('uint8') * 255

        # Save predicted full mask
        cv2.imwrite(str(args.save_dir) + '/' + file.split('/')[-1], mask_final)
