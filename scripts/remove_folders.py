import os
from shutil import rmtree
from pathlib import Path
from tqdm import tqdm

if __name__ == "__main__":
    # Initialize experiment
    base_path = Path('/media/santeri/Transcend/CC_window_rec')
    subdir_to_be_removed = 'Prediction_sweep(1)'
    new_subdir_name = 'Prediction_sweep'

    # List files
    samples = os.listdir(base_path)
    for sample in tqdm(samples, desc='Removing directories'):
        subdir = base_path / sample / subdir_to_be_removed
        new_name = base_path / sample / new_subdir_name
        if subdir.exists():
            #rmtree(subdir)
            subdir.rename(new_name)
        else:
            continue
