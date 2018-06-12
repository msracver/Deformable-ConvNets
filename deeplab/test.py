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

import argparse
import os
import sys
import time
import logging
from config.config import config, update_config

def parse_args():
    parser = argparse.ArgumentParser(description='Test a Deeplab Network')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    # testing
    parser.add_argument('--vis', help='turn on visualization', action='store_true')
    parser.add_argument('--ignore_cache', help='ignore cached results boxes', action='store_true')
    parser.add_argument('--shuffle', help='shuffle data on visualization', action='store_true')
    args = parser.parse_args()
    return args

args = parse_args()
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(curr_path, '../external/mxnet', config.MXNET_VERSION))

import pprint
import mxnet as mx

from symbols import *
from dataset import *
from core.loader import TestDataLoader
from core.tester import Predictor, pred_eval
from utils.load_data import load_gt_segdb, merge_segdb
from utils.load_model import load_param
from utils.create_logger import create_logger

def test_deeplab():
    epoch = config.TEST.test_epoch
    ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]
    image_set = config.dataset.test_image_set
    root_path = config.dataset.root_path
    dataset = config.dataset.dataset
    dataset_path = config.dataset.dataset_path

    logger, final_output_path = create_logger(config.output_path, args.cfg, image_set)
    prefix = os.path.join(final_output_path, '..', '_'.join([iset for iset in config.dataset.image_set.split('+')]), config.TRAIN.model_prefix)

    # print config
    pprint.pprint(config)
    logger.info('testing config:{}\n'.format(pprint.pformat(config)))

    # load symbol and testing data
    sym_instance = eval(config.symbol + '.' + config.symbol)()
    sym = sym_instance.get_symbol(config, is_train=False)

    imdb = eval(dataset)(image_set, root_path, dataset_path, result_path=final_output_path)
    segdb = imdb.gt_segdb()

    # get test data iter
    test_data = TestDataLoader(segdb, config=config, batch_size=len(ctx))

    # infer shape
    data_shape_dict = dict(test_data.provide_data_single)
    sym_instance.infer_shape(data_shape_dict)

    # load model and check parameters
    arg_params, aux_params = load_param(prefix, epoch, process=True)

    sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict, is_train=False)

    # decide maximum shape
    data_names = [k[0] for k in test_data.provide_data_single]
    label_names = ['softmax_label']
    max_data_shape = [[('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]]

    # create predictor
    predictor = Predictor(sym, data_names, label_names,
                          context=ctx, max_data_shapes=max_data_shape,
                          provide_data=test_data.provide_data, provide_label=test_data.provide_label,
                          arg_params=arg_params, aux_params=aux_params)

    # start detection
    pred_eval(predictor, test_data, imdb, vis=args.vis, ignore_cache=args.ignore_cache, logger=logger)

def main():
    print args
    test_deeplab()


if __name__ == '__main__':
    main()
