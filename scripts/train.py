from torch import optim
import argparse
import yaml
import pathlib
import cv2
from functools import partial

from collagen.modelzoo.segmentation import EncoderDecoder
from collagen.losses.segmentation import CombinedLoss, BCEWithLogitsLoss2d, SoftJaccardLoss

from collagen.callbacks.metrics import RunningAverageMeter, JaccardDiceMeter
from collagen.core.utils import auto_detect_device
from collagen.strategies import Strategy
from collagen.savers import ModelSaver

# TODO: Check that this works
# TODO: parse function needs to be changed to binary segmentation!!!
from rabbitccs.training.session import create_data_provider, init_seed, parse_multi_label

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_location', type=pathlib.Path, default='../../../Data/resized_224')
    parser.add_argument('--workdir', type=pathlib.Path, default='../../../workdir/')
    parser.add_argument('--experiment', default='./experiment_conf.yml')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_threads', type=int, default=16)
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--n_epochs', type=int, default=20)
    args = parser.parse_args()
    with open(args.experiment, 'r') as f:
        config = yaml.load(f)

    init_seed(args)

    snapshots_dir = pathlib.Path(args.workdir) / 'snapshots'
    snapshots_dir.mkdir(exist_ok=True)
    device = auto_detect_device()

    data_provider = create_data_provider(args)
    model = EncoderDecoder(**config['model']).to(device)
    loss = CombinedLoss([BCEWithLogitsLoss2d(),
                         SoftJaccardLoss(use_log=config['training']['log_jaccard'])]).to(device)

    optimizer = optim.Adam(model.parameters(),
                           lr=config['training']['lr'],
                           weight_decay=config['training']['wd'])

    train_cbs = (RunningAverageMeter(prefix="train", name="loss"))

    val_cbs = (RunningAverageMeter(prefix="eval", name="loss"),
               JaccardDiceMeter(prefix="eval/lungs", name='jaccard',
                                parse_output=partial(parse_multi_label, cls=0, threshold=0.5),
                                parse_target=lambda x: x[:, 0].squeeze(),
                                class_names=['bg', 'cci']),
               ModelSaver(metric_names='eval/loss',
                          save_dir=str(snapshots_dir),
                          conditions='min', model=model))

    strategy = Strategy(data_provider=data_provider,
                        train_loader_names=tuple(config['data_sampling']['train']['data_provider'].keys()),
                        val_loader_names=tuple(config['data_sampling']['eval']['data_provider'].keys()),
                        data_sampling_config=config['data_sampling'],
                        loss=loss,
                        model=model,
                        n_epochs=args.n_epochs,
                        optimizer=optimizer,
                        train_callbacks=train_cbs,
                        val_callbacks=val_cbs,
                        device=device)

    strategy.run()