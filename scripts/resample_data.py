import os
import numpy as np
from pathlib import Path
from hmdscollagen.training.session import init_experiment
from hmdscollagen.data.utilities import load, save, print_orthogonal
from scipy.ndimage import zoom


if __name__ == "__main__":
    # Initialize experiment
    args, config, device, snapshots_dir, snapshot_name = init_experiment()
    base_path = Path('/data/Repositories/HMDS_orientation/Data/train_rotated/')
    images_loc = base_path / 'hmds'
    images_save = base_path / 'hmds_resampled'

    subdir = 'Manual segmentation'
    images_save.mkdir(exist_ok=True)

    resample = True
    factor = 10
    n_slices = 60

    # Resample large number of slices
    samples = os.listdir(images_loc)
    samples = [name for name in samples if os.path.isdir(os.path.join(images_loc, name))]
    samples.sort()
    for sample in samples:
        if resample:  # Resample slices
            im_path = images_loc / sample

            data = load(str(im_path), axis=(1, 2, 0))
            data = np.flip(data, axis=0)
            data_resampled = zoom(data, (1, 1, 1 / factor), order=0)  # nearest interpolation
            #print_orthogonal(data_resampled)

            save(str(images_save / sample), sample, data_resampled[:, :, :n_slices], dtype='.png')
        else:  # Move segmented samples to training data
            im_path = str(images_loc / sample)
            mask_path = str(images_loc / sample / subdir)
            files = os.listdir(im_path)
            if subdir in files:
                data, _ = load(im_path, axis=(1, 2, 0))
                save(str(images_save / sample), sample, data, dtype='.png')

        #except ValueError:
          #  print(f'Error in sample {sample}')
            #continue
