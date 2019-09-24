import cv2
import numpy as np
import os
from pathlib import Path
import argparse
import dill
import yaml
import pandas as pd
from time import sleep, time

from rabbitccs.data.utilities import load
from rabbitccs.data.visualizations import render_volume

from deeppipeline.segmentation.evaluation.metrics import calculate_iou, calculate_dice, \
    calculate_volumetric_similarity, calculate_confusion_matrix_from_arrays as calculate_conf


cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


if __name__ == "__main__":
    start = time()
    pred_path = Path('/media/dios/dios2/RabbitSegmentation/µCT/predictions_5_fold_trainingset')
    snapshot = Path('dios-erc-gpu_2019_09_18_15_32_33_8samples')

    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_path', type=Path, default='../../../Data/µCT/masks')
    parser.add_argument('--prediction_path', type=Path, default=pred_path)
    parser.add_argument('--save_dir', type=Path, default=pred_path / 'evaluation')
    parser.add_argument('--n_threads', type=int, default=16)
    parser.add_argument('--n_labels', type=int, default=2)
    parser.add_argument('--weight', type=str, choices=['pyramid', 'mean'], default='mean')
    parser.add_argument('--experiment', default='./experiment_config_uCT.yml')
    parser.add_argument('--snapshot', type=Path,
                        default=Path('../../../workdir/snapshots/') / snapshot)
    parser.add_argument('--dtype', type=str, choices=['.bmp', '.png', '.tif'], default='.bmp')
    args = parser.parse_args()

    # Load snapshot configuration
    with open(args.snapshot / 'config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    with open(args.snapshot / 'args.dill', 'rb') as f:
        args_experiment = dill.load(f)

    with open(args.snapshot / 'split_config.dill', 'rb') as f:
        split_config = dill.load(f)

    # Initialize results
    results = {'Sample': [], 'Dice': [], 'IoU': [], 'Similarity': []}

    # Loop for samples
    args.save_dir.mkdir(exist_ok=True)
    samples = os.listdir(args.mask_path)
    samples.sort()
    for sample in samples:
        try:
            sleep(0.5); print(f'==> Processing sample: {sample}')

            # Load image stacks
            mask, files_mask = load(str(args.mask_path / sample), rgb=False, n_jobs=args.n_threads)
            pred, files_pred = load(str(args.prediction_path / sample), rgb=False, n_jobs=args.n_threads)

            conf_matrix = calculate_conf(pred.astype(np.bool), mask.astype(np.bool), args.n_labels)
            dice = calculate_dice(conf_matrix)[1]
            iou = calculate_iou(conf_matrix)[1]
            sim = calculate_volumetric_similarity(conf_matrix)[1]

            print(f'Sample {sample}: dice = {dice}, IoU = {iou}, similarity = {sim}')

            # Save predicted full mask
            render_volume(np.bitwise_xor(pred, mask),
                          savepath=str(args.save_dir / 'visualizations' / (sample + '_difference.png')),
                          white=False, use_outline=False)

            # Update results
            results['Sample'].append(sample)
            results['Dice'].append(dice)
            results['IoU'].append(iou)
            results['Similarity'].append(sim)


        except ValueError:
            print(f'Sample {sample} failing, dimensions not consistent!')
            continue

    # Add average value to
    results['Sample'].append('Average values')
    results['Dice'].append(np.average(results['Dice']))
    results['IoU'].append(np.average(results['IoU']))
    results['Similarity'].append(np.average(results['Similarity']))

    # Write to excel
    writer = pd.ExcelWriter(str(args.save_dir / ('metrics_' + str(snapshot))) + '.xlsx')
    df1 = pd.DataFrame(results)
    df1.to_excel(writer, sheet_name='Metrics')
    writer.save()

    print(f'Metrics evaluated in {(time() - start) // 60} minutes, {(time() - start) % 60} seconds.')
