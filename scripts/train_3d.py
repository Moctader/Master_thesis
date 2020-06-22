import numpy as np
import pathlib
import os.path
from torch import optim, cuda, nn
from time import time
import gc
import cv2
import torchvision
from torchvision.models import resnet18
from hmdscollagen.training.models import SimpleNet
from hmdscollagen.training.net import ReconNet
from hmdscollagen.training.SimpleConvNet import SimpleConvNet
#from hmdscollagen.training.New_model import UNet16
import torch
import torch.nn as nn
import torch.nn.functional as F
from collagen.strategies import Strategy
torch.cuda.is_available()
from scipy import ndimage, misc
from collagen.modelzoo.segmentation import EncoderDecoder
from hmdscollagen.training.session import create_data_provider, init_experiment, init_callbacks, save_transforms,\
    init_loss, parse_grayscale, init_model, save_config

from hmdscollagen.data.splits import build_splits

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
torch.cuda.is_available()

if __name__ == "__main__":
    # Timing
    start = time()
    # Initialize experiment
    args, config, device, snapshots_dir, snapshot_name = init_experiment(experiment='3D')

    # Loss
    loss_criterion = init_loss(config, device=device)

    # Split training folds
    splits_metadata = build_splits(args.data_dir, args, config, parse_grayscale, snapshots_dir, snapshot_name)
    mean, std = splits_metadata['mean'], splits_metadata['std']

    # Save transforms list
    save_transforms(snapshots_dir / snapshot_name, config, args, mean, std)


    # Training for separate folds
    for fold in range(config['training']['n_folds']):
        print(f'\nTraining fold {fold}')
        # Initialize data provider
        data_provider = create_data_provider(args, config, parse_grayscale, metadata=splits_metadata[f'fold_{fold}'],
                                             mean=mean, std=std)
        #model= EncoderDecoder(**config['model']).to(device)
        #model = ReconNet(**config['model']).to(device)
        #model = init_model(config['model_selection'])
        #model = torchvision.models.resnet18(pretrained=True)
        model = SimpleConvNet().to(device)
       # model = SimpleNet().to(device)

        # Optimizer
        optimizer = optim.Adam(model.parameters(),
                               lr=config['training']['lr'],
                               weight_decay=config['training']['wd'])
        # Callbacks
        train_cbs, val_cbs = init_callbacks(fold, config, snapshots_dir, snapshot_name, model, optimizer, data_provider,
                                            mean, std)
        # Run training
        strategy = Strategy(data_provider=data_provider,
                            train_loader_names=tuple(config['data_sampling']['train']['data_provider'].keys()),
                            val_loader_names=tuple(config['data_sampling']['eval']['data_provider'].keys()),
                            data_sampling_config=config['data_sampling'],
                            loss=loss_criterion,
                            model=model,
                            n_epochs=args.n_epochs,
                            optimizer=optimizer,
                            train_callbacks=train_cbs,
                            val_callbacks=val_cbs,
                            device=device)
        strategy.run()
       # save_config(snapshots_dir / snapshot_name, config, args, model)
        #Manage memory
        del strategy


        #os.chdir('/data/Repositories/HMDS_collagen/workdir/snapshots/mipt-stud-dl-b_2020_02_18_11_06_55/')
        #FILE='model.pth'
        Save_model=torch.save(model.state_dict(), FILE)
        #savepath = '/data/Repositories/HMDS_collagen/workdir/snapshots/'
        #save=os.path.join(str(savepath, Save_model)
        del model
        cuda.empty_cache()
        gc.collect()

    dur = time() - start
    print(f'Model trained in {dur // 3600} hours, {(dur % 3600) // 60} minutes, {dur % 60} seconds.')
