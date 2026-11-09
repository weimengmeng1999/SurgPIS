# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from detectron2.config import CfgNode as CN

def add_wsl_config(cfg):
    # ---------------------------------------------------------------------------- #
    # WSL options
    # ---------------------------------------------------------------------------- #
    _C = cfg
    _C.WSL = CN()
    _C.WSL.TRAIN_WSL = True
    _C.WSL.TEACHER_CKPT = ""
    _C.WSL.BURNIN_ITER = 0
    _C.WSL.PERCENTAGE = 100
    _C.WSL.FREQ = 1 #3
    _C.WSL.EMA_DECAY = 0.9996
    _C.WSL.CKPT_TARGET = 'TEACHER'
    _C.WSL.EVAL_WHO = "STUDENT"
    _C.WSL.WEIGHTS = ""
    _C.WSL.WEIGHT_IIS = 2.0
    _C.WSL.WEIGHT_PSS = 3.0
    # dataset names for the two unlabeled streams -- overridable so a
    # diagnostic run can point at a single-image dataset instead.
    _C.WSL.IIS_DATASET_TRAIN = "endovis17_iis_train"
    _C.WSL.PSS_DATASET_TRAIN = "sar_rarp50_pss_train"