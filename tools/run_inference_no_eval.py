# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Standalone diagnostic runner: builds the model, loads a checkpoint, and
runs inference over cfg.DATASETS.TEST with no evaluator at all -- just to
exercise SAVE_PREDICTIONS and get saved label-map PNGs out, without needing
the dataset's registered evaluator_type to actually match this model's
output format (e.g. sar_rarp50_pss_train is registered as "sem_seg" for its
own training-time semantic mapper, but this model's semantic-inference
output is over a completely different, instrument-level class space).

Usage: same config-file/opts as train_net.py --eval-only, just via this
script instead so the evaluator-building step is skipped entirely.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_argument_parser, launch
from detectron2.evaluation import inference_on_dataset

from modules.ensemble import EnsembleTSModel
from train_net import Trainer, setup


def main(args):
    cfg = setup(args)
    model = Trainer.build_model(cfg)
    # Stage-1 checkpoints are saved via EnsembleTSModel(model, None), so their
    # weights live under the "modelTeacher." prefix -- wrap the same way here
    # rather than loading into the bare model directly.
    DetectionCheckpointer(EnsembleTSModel(model, None)).load(cfg.MODEL.WEIGHTS)
    model.eval()

    for dataset_name in cfg.DATASETS.TEST:
        data_loader = Trainer.build_test_loader(cfg, dataset_name)
        inference_on_dataset(model, data_loader, evaluator=None)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
