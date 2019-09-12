import numpy as np
from torch import optim, cuda, nn
import gc
import cv2

from collagen.modelzoo.segmentation import EncoderDecoder
from collagen.losses.segmentation import CombinedLoss, BCEWithLogitsLoss2d, SoftJaccardLoss
from collagen.strategies import Strategy

from rabbitccs.training.session import create_data_provider, init_experiment, init_callbacks, save_transforms,\
    init_loss, parse_grayscale, parse_color_im

from rabbitccs.data.splits import build_splits

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


if __name__ == "__main__":

    # Initialize experiment
    args, config, device, snapshots_dir, snapshot_name = init_experiment(experiment='3D')

    # Loss
    loss_criterion = init_loss(config, device=device)

    # Split training folds
    splits_metadata = build_splits(args.data_location, args, config, parse_color_im, snapshots_dir, snapshot_name)
    mean, std = splits_metadata['mean'], splits_metadata['std']

    # Save transforms list
    save_transforms(snapshots_dir / snapshot_name, config, args, mean, std)

    # Initialize results
    kfold_train_losses = []
    kfold_val_losses = []
    kfold_val_accuracies = []

    # Training for separate folds
    for fold in range(config['training']['n_folds']):
        print(f'\nTraining fold {fold}')

        data_provider = create_data_provider(args, config, parse_color_im, metadata=splits_metadata[f'fold_{fold}'],
                                             mean=mean, std=std)
        model = EncoderDecoder(**config['model']).to(device)
        model = nn

        optimizer = optim.Adam(model.parameters(),
                               lr=config['training']['lr'],
                               weight_decay=config['training']['wd'])

        train_cbs, val_cbs = init_callbacks(fold, config, snapshots_dir, snapshot_name, model, optimizer, data_provider,
                                            mean, std)

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
        kfold_train_losses.append(train_cbs[0].current())
        kfold_val_losses.append(val_cbs[0].current())
        del strategy
        del model
        cuda.empty_cache()
        gc.collect()

    print("k-fold training loss: {}".format(np.asarray(kfold_train_losses).mean()))
    print("k-fold validation loss: {}".format(np.asarray(kfold_val_losses).mean()))

