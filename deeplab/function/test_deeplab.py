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

import argparse
import pprint
import logging
import time
import os
import mxnet as mx

from config.config import config, generate_config, update_config
from config.dataset_conf import dataset
from config.network_conf import network
from symbols import *
from dataset import *
from core.loader import TestDataLoader
from core.tester import Predictor, pred_eval
from utils.load_model import load_param

def test_deeplab(network, dataset, image_set, root_path, dataset_path,
              ctx, prefix, epoch,
              vis, logger=None, output_path=None):
    if not logger:
        assert False, 'require a logger'

    # print config
    pprint.pprint(config)
    logger.info('testing config:{}\n'.format(pprint.pformat(config)))

    # load symbol and testing data
    sym = eval('get_' + network + '_test')(num_classes=config.dataset.NUM_CLASSES)
    imdb = eval(dataset)(image_set, root_path, dataset_path, result_path=output_path)
    segdb = imdb.gt_segdb()

    # get test data iter
    test_data = TestDataLoader(segdb, batch_size=len(ctx))

    # load model
    # arg_params, aux_params = load_param(prefix, epoch, convert=True, ctx=ctx, process=True)
    arg_params, aux_params = load_param(prefix, epoch, process=True)

    # infer shape
    data_shape_dict = dict(test_data.provide_data_single)
    arg_shape, _, aux_shape = sym.infer_shape(**data_shape_dict)
    arg_shape_dict = dict(zip(sym.list_arguments(), arg_shape))
    aux_shape_dict = dict(zip(sym.list_auxiliary_states(), aux_shape))

    # check parameters
    for k in sym.list_arguments():
        if k in data_shape_dict or k in ['softmax_label']:
            continue
        assert k in arg_params, k + ' not initialized'
        assert arg_params[k].shape == arg_shape_dict[k], \
            'shape inconsistent for ' + k + ' inferred ' + str(arg_shape_dict[k]) + ' provided ' + str(arg_params[k].shape)
    for k in sym.list_auxiliary_states():
        assert k in aux_params, k + ' not initialized'
        assert aux_params[k].shape == aux_shape_dict[k], \
            'shape inconsistent for ' + k + ' inferred ' + str(aux_shape_dict[k]) + ' provided ' + str(aux_params[k].shape)

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
    pred_eval(predictor, test_data, imdb, vis=vis, logger=logger)

