# -*- coding: utf-8 -*-
# Copyright 2021 Minggui Song.
# All rights reserved.
#
from easydict import EasyDict as edict

config = edict()

config.TRAIN = edict()
config.TRAIN.input_dir = './Data/m7G/processed/'
config.TRAIN.inst_len = 50
config.TRAIN.inst_stride = 20
config.TRAIN.model_name = 'WeakRM'
config.TRAIN.fusion_method = 'Max'
config.TRAIN.epoch = 50
config.TRAIN.lr_init = 1e-4
config.TRAIN.lr_decay = 1e-5
config.TRAIN.CLASSES = 4

config.cropping = False
config.nt_crop = 400

config.eval_after_train = False
config.threshold = 0.5

config.sample_names = ['gar']
#config.sample_names = ['ath']