# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Zheng Zhang
# --------------------------------------------------------
# Based on:
# MX-RCNN
# Copyright (c) 2016 by Contributors
# Licence under The Apache 2.0 License
# https://github.com/ijkguo/mx-rcnn/
# --------------------------------------------------------

import yaml
import numpy as np
from easydict import EasyDict as edict

config = edict()

config.MXNET_VERSION = ''
config.output_path = ''
config.symbol = ''
config.gpus = ''
config.CLASS_AGNOSTIC = True
config.SCALES = [(360, 600)]  # first is scale (the shorter side); second is max size

# default training
config.default = edict()
config.default.frequent = 1000
config.default.kvstore = 'device'

# network related params
config.network = edict()
config.network.pretrained = '../model/pretrained_model/resnet_v1-101'
config.network.pretrained_epoch = 0
config.network.PIXEL_MEANS = np.array([103.06, 115.90, 123.15])
config.network.IMAGE_STRIDE = 0
config.network.FIXED_PARAMS = ['conv1', 'bn_conv1', 'res2', 'bn2', 'gamma', 'beta']

# dataset related params
config.dataset = edict()
config.dataset.dataset = 'cityscapes'
config.dataset.image_set = 'leftImg8bit_train'
config.dataset.test_image_set = 'leftImg8bit_val'
config.dataset.root_path = '../data'
config.dataset.dataset_path = '../data/cityscapes'
config.dataset.NUM_CLASSES = 19
config.dataset.annotation_prefix = 'gtFine'

config.TRAIN = edict()
config.TRAIN.lr = 0
config.TRAIN.lr_step = ''
config.TRAIN.warmup = False
config.TRAIN.warmup_lr = 0
config.TRAIN.warmup_step = 0
config.TRAIN.momentum = 0.9
config.TRAIN.wd = 0.0005
config.TRAIN.begin_epoch = 0
config.TRAIN.end_epoch = 0
config.TRAIN.model_prefix = 'deeplab'

# whether resume training
config.TRAIN.RESUME = False
# whether flip image
config.TRAIN.FLIP = True
# whether shuffle image
config.TRAIN.SHUFFLE = True
# whether use OHEM
config.TRAIN.ENABLE_OHEM = False
# size of images for each device, 2 for rcnn, 1 for rpn and e2e
config.TRAIN.BATCH_IMAGES = 1

config.TEST = edict()
# size of images for each device
config.TEST.BATCH_IMAGES = 1

# Test Model Epoch
config.TEST.test_epoch = 0

def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    if k == 'TRAIN':
                        if 'BBOX_WEIGHTS' in v:
                            v['BBOX_WEIGHTS'] = np.array(v['BBOX_WEIGHTS'])
                    elif k == 'network':
                        if 'PIXEL_MEANS' in v:
                            v['PIXEL_MEANS'] = np.array(v['PIXEL_MEANS'])
                    for vk, vv in v.items():
                        config[k][vk] = vv
                else:
                    if k == 'SCALES':
                        config[k][0] = (tuple(v))
                    else:
                        config[k] = v
            else:
                raise ValueError("key must exist in config.py")
