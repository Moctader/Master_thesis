#!/usr/bin/env bash

ROOT=/media/santeri/data/RabbitCCS/
EXPERIMENTS=experiments/

python scripts/experiment_runner.py --data_root ${ROOT}/Data \
                                    --workdir ${ROOT}/workdir \
                                    --experiments ${EXPERIMENTS} \
                                    --log_dir ${ROOT}/workdir/experiment_logs \
                                    --script_path scripts/train.py

for SNP in $(ls ${ROOT}/workdir/snapshots | grep "2019_")
do
    python scripts/inference_tiles.py --workdir ${ROOT}/workdir --snapshot ${SNP}
done