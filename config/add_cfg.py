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
    _C.WSL.BURNIN_ITER = 10000
    _C.WSL.FREQ = 1 #3
    _C.WSL.EMA_DECAY = 0.9996
    _C.WSL.CKPT_TARGET = 'TEACHER'
    _C.WSL.EVAL_WHO = "STUDENT"
    _C.WSL.WEIGHTS = ""