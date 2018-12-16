# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Yuwen Xiong, Bin Xiao
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
config.SCALES = [(600, 1000)]  # first is scale (the shorter side); second is max size

# default training
config.default = edict()
config.default.frequent = 20
config.default.kvstore = 'device'

# network related params
config.network = edict()
config.network.pretrained = ''
config.network.pretrained_epoch = 0
config.network.PIXEL_MEANS = np.array([0, 0, 0])
config.network.IMAGE_STRIDE = 0
config.network.RPN_FEAT_STRIDE = 16
config.network.RCNN_FEAT_STRIDE = 16
config.network.FIXED_PARAMS = ['gamma', 'beta']
config.network.FIXED_PARAMS_SHARED = ['gamma', 'beta']
config.network.ANCHOR_SCALES = (8, 16, 32)
config.network.ANCHOR_RATIOS = (0.5, 1, 2)
config.network.NUM_ANCHORS = len(config.network.ANCHOR_SCALES) * len(config.network.ANCHOR_RATIOS)

# dataset related params
config.dataset = edict()
config.dataset.dataset = 'PascalVOC'
config.dataset.image_set = '2007_trainval'
config.dataset.test_image_set = '2007_test'
config.dataset.root_path = './data'
config.dataset.dataset_path = './data/VOCdevkit'
config.dataset.NUM_CLASSES = 21


config.TRAIN = edict()

config.TRAIN.lr = 0
config.TRAIN.lr_step = ''
config.TRAIN.lr_factor = 0.1
config.TRAIN.warmup = False
config.TRAIN.warmup_lr = 0
config.TRAIN.warmup_step = 0
config.TRAIN.momentum = 0.9
config.TRAIN.wd = 0.0005
config.TRAIN.begin_epoch = 0
config.TRAIN.end_epoch = 0
config.TRAIN.model_prefix = ''

config.TRAIN.ALTERNATE = edict()
config.TRAIN.ALTERNATE.RPN_BATCH_IMAGES = 0
config.TRAIN.ALTERNATE.RCNN_BATCH_IMAGES = 0
config.TRAIN.ALTERNATE.rpn1_lr = 0
config.TRAIN.ALTERNATE.rpn1_lr_step = ''    # recommend '2'
config.TRAIN.ALTERNATE.rpn1_epoch = 0       # recommend 3
config.TRAIN.ALTERNATE.rfcn1_lr = 0
config.TRAIN.ALTERNATE.rfcn1_lr_step = ''   # recommend '5'
config.TRAIN.ALTERNATE.rfcn1_epoch = 0      # recommend 8
config.TRAIN.ALTERNATE.rpn2_lr = 0
config.TRAIN.ALTERNATE.rpn2_lr_step = ''    # recommend '2'
config.TRAIN.ALTERNATE.rpn2_epoch = 0       # recommend 3
config.TRAIN.ALTERNATE.rfcn2_lr = 0
config.TRAIN.ALTERNATE.rfcn2_lr_step = ''   # recommend '5'
config.TRAIN.ALTERNATE.rfcn2_epoch = 0      # recommend 8
# optional
config.TRAIN.ALTERNATE.rpn3_lr = 0
config.TRAIN.ALTERNATE.rpn3_lr_step = ''    # recommend '2'
config.TRAIN.ALTERNATE.rpn3_epoch = 0       # recommend 3

# whether resume training
config.TRAIN.RESUME = False
# whether flip image
config.TRAIN.FLIP = True
# whether shuffle image
config.TRAIN.SHUFFLE = True
# whether use OHEM
config.TRAIN.ENABLE_OHEM = False
# size of images for each device, 2 for rcnn, 1 for rpn and e2e
config.TRAIN.BATCH_IMAGES = 2
# e2e changes behavior of anchor loader and metric
config.TRAIN.END2END = False
# group images with similar aspect ratio
config.TRAIN.ASPECT_GROUPING = True

# R-CNN
# rcnn rois batch size
config.TRAIN.BATCH_ROIS = 128
config.TRAIN.BATCH_ROIS_OHEM = 128
# rcnn rois sampling params
config.TRAIN.FG_FRACTION = 0.25
config.TRAIN.FG_THRESH = 0.5
config.TRAIN.BG_THRESH_HI = 0.5
config.TRAIN.BG_THRESH_LO = 0.0
# rcnn bounding box regression params
config.TRAIN.BBOX_REGRESSION_THRESH = 0.5
config.TRAIN.BBOX_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0])

# RPN anchor loader
# rpn anchors batch size
config.TRAIN.RPN_BATCH_SIZE = 256
# rpn anchors sampling params
config.TRAIN.RPN_FG_FRACTION = 0.5
config.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
config.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
config.TRAIN.RPN_CLOBBER_POSITIVES = False
# rpn bounding box regression params
config.TRAIN.RPN_BBOX_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
config.TRAIN.RPN_POSITIVE_WEIGHT = -1.0

# used for end2end training
# RPN proposal
config.TRAIN.CXX_PROPOSAL = True
config.TRAIN.RPN_NMS_THRESH = 0.7
config.TRAIN.RPN_PRE_NMS_TOP_N = 12000
config.TRAIN.RPN_POST_NMS_TOP_N = 2000
config.TRAIN.RPN_MIN_SIZE = config.network.RPN_FEAT_STRIDE
# approximate bounding box regression
config.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED = False
config.TRAIN.BBOX_MEANS = (0.0, 0.0, 0.0, 0.0)
config.TRAIN.BBOX_STDS = (0.1, 0.1, 0.2, 0.2)

config.TEST = edict()

# R-CNN testing
# use rpn to generate proposal
config.TEST.HAS_RPN = False
# size of images for each device
config.TEST.BATCH_IMAGES = 1

# RPN proposal
config.TEST.CXX_PROPOSAL = True
config.TEST.RPN_NMS_THRESH = 0.7
config.TEST.RPN_PRE_NMS_TOP_N = 6000
config.TEST.RPN_POST_NMS_TOP_N = 300
config.TEST.RPN_MIN_SIZE = config.network.RPN_FEAT_STRIDE

# RPN generate proposal
config.TEST.PROPOSAL_NMS_THRESH = 0.7
config.TEST.PROPOSAL_PRE_NMS_TOP_N = 20000
config.TEST.PROPOSAL_POST_NMS_TOP_N = 2000
config.TEST.PROPOSAL_MIN_SIZE = config.network.RPN_FEAT_STRIDE

# RCNN nms
config.TEST.NMS = 0.3

config.TEST.max_per_image = 300

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
