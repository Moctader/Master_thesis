import numpy as np
from torch import optim
import cv2
from functools import partial

from collagen.modelzoo.segmentation import EncoderDecoder
from collagen.losses.segmentation import CombinedLoss, BCEWithLogitsLoss2d, SoftJaccardLoss
from collagen.strategies import Strategy

# TODO: parse function needs to be changed to binary segmentation!!!
from rabbitccs.training.session import create_data_provider, init_experiment, init_callbacks
from rabbitccs.data.splits import build_splits


cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


if __name__ == "__main__":

    # Initialize experiment
    args, config, device, snapshots_dir, snapshot_name, logs_dir = init_experiment()

    loss_criterion = CombinedLoss([BCEWithLogitsLoss2d(),
                                   SoftJaccardLoss(use_log=config['training']['log_jaccard'])]).to(device)

    # Split training folds
    splits_list, mean, std = build_splits(args.data_location, args, config, snapshots_dir)

    # Initialize results
    kfold_train_losses = []
    kfold_val_losses = []
    kfold_val_accuracies = []

    # Training for separate folds
    for fold in range(len(splits_list)):
        print(f'\nTraining fold {fold}')

        data_provider = create_data_provider(args, config, metadata=splits_list[fold], mean=mean, std=std)
        model = EncoderDecoder(**config['model']).to(device)

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

    print("k-fold training loss: {}".format(np.asarray(kfold_train_losses).mean()))
    print("k-fold validation loss: {}".format(np.asarray(kfold_val_losses).mean()))
