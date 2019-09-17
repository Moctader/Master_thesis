import numpy as np
import os
import cv2
from tqdm import tqdm
from joblib import Parallel, delayed


def load_images(path, n_jobs=12, rgb=False, uCT=False):
    """
    Loads multiple images from directory and stacks them into 3D numpy array

    Parameters
    ----------
    path : str
        Path to image stack.
    axis : tuple
        Order of loaded sample axes.
    n_jobs : int
        Number of parallel workers. Check N of CPU cores.
    Returns
    -------
    Loaded stack of images as 3D numpy array.
    """

    # List all files in alphabetic order
    files = os.listdir(path)
    files.sort()

    # Exclude extra files
    newlist = []
    if uCT:
        for file in files:
            if file.endswith('.png') or file.endswith('.bmp') or file.endswith('.tif'):
                try:
                    int(file[-7:-4])
                    newlist.append(file)
                except ValueError:
                    continue
    else:
        for file in files:
            if file.endswith('.png') or file.endswith('.bmp') or file.endswith('.tif'):
                newlist.append(file)
    files = newlist[:]  # replace list
    files.sort()

    # Load images
    if rgb:
        data = Parallel(n_jobs=n_jobs)(delayed(read_image_rgb)(path, file) for file in tqdm(files, 'Loading'))
        return files, data
    else:
        data = Parallel(n_jobs=n_jobs)(delayed(read_image_gray)(path, file) for file in tqdm(files, 'Loading'))
        return files, data


def load(path, axis=(0, 1, 2), n_jobs=12, rgb=False):
    """
    Loads an image stack as numpy array.

    Parameters
    ----------
    path : str
        Path to image stack.
    axis : tuple
        Order of loaded sample axes.
    n_jobs : int
        Number of parallel workers. Check N of CPU cores.
    Returns
    -------
    Loaded stack as 3D numpy array.
    """
    files = os.listdir(path)
    files.sort()
    # Exclude extra files
    newlist = []
    for file in files:
        if file.endswith('.png') or file.endswith('.bmp') or file.endswith('.tif'):
            try:
                int(file[-7:-4])
                newlist.append(file)
            except ValueError:
                continue
    files = newlist[:]  # replace list
    # Load images
    if rgb:
        data = Parallel(n_jobs=n_jobs)(delayed(read_image_rgb)(path, file) for file in tqdm(files, 'Loading'))
    else:
        data = Parallel(n_jobs=n_jobs)(delayed(read_image_gray)(path, file) for file in tqdm(files, 'Loading'))
    # Transpose array
    if axis != (0, 1, 2):
        return np.transpose(np.array(data), axis)

    return np.array(data)


def read_image_gray(path, file):
    """Reads image from given path."""
    # Image
    f = os.path.join(path, file)
    image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    return image


def read_image_rgb(path, file):
    """Reads image from given path."""
    # Image
    f = os.path.join(path, file)
    image = cv2.imread(f, cv2.IMREAD_COLOR)

    return image


def save(path, file_name, data, n_jobs=12, dtype='.png'):
    """
    Save a volumetric 3D dataset in given directory.

    Parameters
    ----------
    path : str
        Directory for dataset.
    file_name : str
        Prefix for the image filenames.
    data : 3D numpy array
        Volumetric data to be saved.
    n_jobs : int
        Number of parallel workers. Check N of CPU cores.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    nfiles = np.shape(data)[2]

    if data[0, 0, 0].dtype is bool:
        data = data * 255

    # Parallel saving (nonparallel if n_jobs = 1)
    Parallel(n_jobs=n_jobs)(delayed(cv2.imwrite)
                            (path + '/' + file_name + '_' + str(k).zfill(8) + dtype,
                             data[:, :, k].squeeze().astype('uint8'))
                            for k in tqdm(range(nfiles), 'Saving dataset'))


def save_images(path, file_names, data, n_jobs=12, crop=False):
    """
    Save a set of RGB images in given directory.

    Parameters
    ----------
    path : str
        Directory for dataset.
    file_name : str
        Prefix for the image filenames.
    data : 3D numpy array
        Volumetric data to be saved.
    n_jobs : int
        Number of parallel workers. Check N of CPU cores.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    nfiles = len(data)

    # Parallel saving (nonparallel if n_jobs = 1)
    if crop:
        Parallel(n_jobs=n_jobs)(delayed(cv2.imwrite)
                                (path + '/' + file_names[k][:-4] + '.png', data[k][:512, :])
                                for k in tqdm(range(nfiles), 'Saving images'))
    else:
        Parallel(n_jobs=n_jobs)(delayed(cv2.imwrite)
                                (path + '/' + file_names[k][:-4] + '.png', data[k][:, :])
                                for k in tqdm(range(nfiles), 'Saving images'))


def bounding_box(image, largest=True):
    """
    Return bounding box and contours of a mask.

    Parameters
    ----------
    image : 2D numpy array
        Input mask
    largest : bool
        Option to return only the largest contour. All contours returned otherwise.
    Returns
    -------
    Bounding box coordinates (tuple) and list of contours (or largest contour).
    """
    # Threshold and create Mat
    _, mask = cv2.threshold(image, thresh=0.5, maxval=1, type=0)
    # All contours
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # Largest contour
    c = max(contours, key=cv2.contourArea)

    # Return bounding rectangle for largest contour
    if largest:
        return cv2.boundingRect(c), c
    else:
        return cv2.boundingRect(c), contours


def mask2rle(img, width, height):
    rle = []
    last_color = 0
    current_pixel = 0
    run_start = -1
    run_length = 0

    for x in range(width):
        for y in range(height):
            current_color = img[x][y]
            if current_color != last_color:
                if current_color == 255:
                    run_start = current_pixel
                    run_length = 1
                else:
                    rle.append(str(run_start))
                    rle.append(str(run_length))
                    run_start = -1
                    run_length = 0
                    current_pixel = 0
            elif run_start > -1:
                run_length += 1
            last_color = current_color
            current_pixel += 1

    return " ".join(rle)
