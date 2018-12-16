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

import _init_paths

import time
import argparse
import logging
import pprint
import os
import sys
from config.config import config, update_config

def parse_args():
    parser = argparse.ArgumentParser(description='Train deeplab network')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent', help='frequency of logging', default=config.default.frequent, type=int)
    args = parser.parse_args()
    return args

args = parse_args()
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(curr_path, '../external/mxnet', config.MXNET_VERSION))


import shutil
import numpy as np
import mxnet as mx

from symbols import *
from core import callback, metric
from core.loader import TrainDataLoader
from core.module import MutableModule
from utils.load_data import load_gt_segdb, merge_segdb
from utils.load_model import load_param
from utils.PrefetchingIter import PrefetchingIter
from utils.create_logger import create_logger
from utils.lr_scheduler import WarmupMultiFactorScheduler

def train_net(args, ctx, pretrained, epoch, prefix, begin_epoch, end_epoch, lr, lr_step):
    logger, final_output_path = create_logger(config.output_path, args.cfg, config.dataset.image_set)
    prefix = os.path.join(final_output_path, prefix)

    # load symbol
    shutil.copy2(os.path.join(curr_path, 'symbols', config.symbol + '.py'), final_output_path)
    sym_instance = eval(config.symbol + '.' + config.symbol)()
    sym = sym_instance.get_symbol(config, is_train=True)
    #sym = eval('get_' + args.network + '_train')(num_classes=config.dataset.NUM_CLASSES)

    # setup multi-gpu
    batch_size = len(ctx)
    input_batch_size = config.TRAIN.BATCH_IMAGES * batch_size

    # print config
    pprint.pprint(config)
    logger.info('training config:{}\n'.format(pprint.pformat(config)))

    # load dataset and prepare imdb for training
    image_sets = [iset for iset in config.dataset.image_set.split('+')]
    segdbs = [load_gt_segdb(config.dataset.dataset, image_set, config.dataset.root_path, config.dataset.dataset_path,
                            result_path=final_output_path, flip=config.TRAIN.FLIP)
              for image_set in image_sets]
    segdb = merge_segdb(segdbs)

    # load training data
    train_data = TrainDataLoader(sym, segdb, config, batch_size=input_batch_size, crop_height=config.TRAIN.CROP_HEIGHT, crop_width=config.TRAIN.CROP_WIDTH,
                                 shuffle=config.TRAIN.SHUFFLE, ctx=ctx)

    # infer max shape
    max_scale = [(config.TRAIN.CROP_HEIGHT, config.TRAIN.CROP_WIDTH)]
    max_data_shape = [('data', (config.TRAIN.BATCH_IMAGES, 3, max([v[0] for v in max_scale]), max([v[1] for v in max_scale])))]
    max_label_shape = [('label', (config.TRAIN.BATCH_IMAGES, 1, max([v[0] for v in max_scale]), max([v[1] for v in max_scale])))]
    # max_data_shape, max_label_shape = train_data.infer_shape(max_data_shape, max_label_shape)
    print 'providing maximum shape', max_data_shape, max_label_shape

    # infer shape
    data_shape_dict = dict(train_data.provide_data_single + train_data.provide_label_single)
    pprint.pprint(data_shape_dict)
    sym_instance.infer_shape(data_shape_dict)

    # load and initialize params
    if config.TRAIN.RESUME:
        print 'continue training from ', begin_epoch
        arg_params, aux_params = load_param(prefix, begin_epoch, convert=True)
    else:
        print pretrained
        arg_params, aux_params = load_param(pretrained, epoch, convert=True)
        sym_instance.init_weights(config, arg_params, aux_params)

    # check parameter shapes
    sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict)

    # create solver
    fixed_param_prefix = config.network.FIXED_PARAMS
    data_names = [k[0] for k in train_data.provide_data_single]
    label_names = [k[0] for k in train_data.provide_label_single]

    mod = MutableModule(sym, data_names=data_names, label_names=label_names,
                        logger=logger, context=ctx, max_data_shapes=[max_data_shape for _ in xrange(batch_size)],
                        max_label_shapes=[max_label_shape for _ in xrange(batch_size)], fixed_param_prefix=fixed_param_prefix)

    # decide training params
    # metric
    fcn_loss_metric = metric.FCNLogLossMetric(config.default.frequent * batch_size)
    eval_metrics = mx.metric.CompositeEvalMetric()

    # rpn_eval_metric, rpn_cls_metric, rpn_bbox_metric, eval_metric, cls_metric, bbox_metric
    for child_metric in [fcn_loss_metric]:
        eval_metrics.add(child_metric)

    # callback
    batch_end_callback = callback.Speedometer(train_data.batch_size, frequent=args.frequent)
    epoch_end_callback = mx.callback.module_checkpoint(mod, prefix, period=1, save_optimizer_states=True)

    # decide learning rate
    base_lr = lr
    lr_factor = 0.1
    lr_epoch = [float(epoch) for epoch in lr_step.split(',')]
    lr_epoch_diff = [epoch - begin_epoch for epoch in lr_epoch if epoch > begin_epoch]
    lr = base_lr * (lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
    lr_iters = [int(epoch * len(segdb) / batch_size) for epoch in lr_epoch_diff]
    print 'lr', lr, 'lr_epoch_diff', lr_epoch_diff, 'lr_iters', lr_iters

    lr_scheduler = WarmupMultiFactorScheduler(lr_iters, lr_factor, config.TRAIN.warmup, config.TRAIN.warmup_lr, config.TRAIN.warmup_step)

    # optimizer
    optimizer_params = {'momentum': config.TRAIN.momentum,
                        'wd': config.TRAIN.wd,
                        'learning_rate': lr,
                        'lr_scheduler': lr_scheduler,
                        'rescale_grad': 1.0,
                        'clip_gradient': None}

    if not isinstance(train_data, PrefetchingIter):
        train_data = PrefetchingIter(train_data)

    # train
    mod.fit(train_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback,
            batch_end_callback=batch_end_callback, kvstore=config.default.kvstore,
            optimizer='sgd', optimizer_params=optimizer_params,
            arg_params=arg_params, aux_params=aux_params, begin_epoch=begin_epoch, num_epoch=end_epoch)

def main():
    print 'Called with argument:', args
    ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]
    train_net(args, ctx, config.network.pretrained, config.network.pretrained_epoch, config.TRAIN.model_prefix,
              config.TRAIN.begin_epoch, config.TRAIN.end_epoch, config.TRAIN.lr, config.TRAIN.lr_step)

if __name__ == '__main__':
    main()
