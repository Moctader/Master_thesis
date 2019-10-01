import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from glob import glob
from tqdm import tqdm
import argparse
import pandas as pd
from time import time

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


if __name__ == "__main__":
    start = time()


    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_path', type=Path, default='../../../Data/masks')
    parser.add_argument('--save_dir', type=Path, default='../../../Data/masks_smoothed')
    parser.add_argument('--k_closing', type=tuple, default=(13, 13))
    parser.add_argument('--k_gauss', type=tuple, default=(9, 9))
    parser.add_argument('--k_median', type=int, default=7)
    parser.add_argument('--n_threads', type=int, default=16)
    parser.add_argument('--plot', type=bool, default=False)
    parser.add_argument('--dtype', type=str, choices=['.bmp', '.png', '.tif'], default='.bmp')
    args = parser.parse_args()

    # Loop for samples
    args.save_dir.mkdir(exist_ok=True)
    #samples = os.listdir(str(args.mask_path))
    samples = [os.path.basename(x) for x in glob(str(args.mask_path / '*.png'))]
    samples.sort()
    for sample in tqdm(samples, 'Smoothing'):
        #try:
        # Load image
        img = cv2.imread(str(args.mask_path / sample), cv2.IMREAD_GRAYSCALE)
        if args.plot:
            plt.imshow(img); plt.title('Loaded image'); plt.show()
        # Opening
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=args.k_closing)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel=kernel)
        #plt.imshow(img); plt.title('Closing'); plt.show()

        # Gaussian blur
        img = cv2.GaussianBlur(img, ksize=args.k_gauss, sigmaX=0, sigmaY=0)
        #plt.imshow(img); plt.title('Gaussian blur'); plt.show()
        # Median filter (round kernel 7)
        img = cv2.medianBlur(img, ksize=args.k_median)
        #plt.imshow(img); plt.title('Median filter'); plt.show()
        # Threshold >= 125
        img = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)[1]
        if args.plot:
            plt.imshow(img); plt.title('Threshold'); plt.show()
        # Save image
        cv2.imwrite(str(args.save_dir / sample), img)
        #except Exception as e:
        #    print(f'Sample {sample} failing due to error:\n\n{e}\n!')
        #    continue

    print(f'Metrics evaluated in {(time() - start) // 60} minutes, {(time() - start) % 60} seconds.')
