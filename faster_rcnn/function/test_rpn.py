# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Yuwen Xiong
# --------------------------------------------------------

import argparse
import pprint
import logging
import mxnet as mx

from symbols import *
from dataset import *
from core.loader import TestLoader
from core.tester import Predictor, generate_proposals
from utils.load_model import load_param


def test_rpn(cfg, dataset, image_set, root_path, dataset_path,
             ctx, prefix, epoch,
             vis, shuffle, thresh, logger=None, output_path=None):
    # set up logger
    if not logger:
        logging.basicConfig()
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

    # rpn generate proposal cfg
    cfg.TEST.HAS_RPN = True

    # print cfg
    pprint.pprint(cfg)
    logger.info('testing rpn cfg:{}\n'.format(pprint.pformat(cfg)))

    # load symbol
    sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
    sym = sym_instance.get_symbol_rpn(cfg, is_train=False)

    # load dataset and prepare imdb for training
    imdb = eval(dataset)(image_set, root_path, dataset_path, result_path=output_path)
    roidb = imdb.gt_roidb()
    test_data = TestLoader(roidb, cfg, batch_size=len(ctx), shuffle=shuffle, has_rpn=True)

    # load model
    arg_params, aux_params = load_param(prefix, epoch)

    # infer shape
    data_shape_dict = dict(test_data.provide_data_single)
    sym_instance.infer_shape(data_shape_dict)

    # check parameters
    sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict, is_train=False)

    # decide maximum shape
    data_names = [k[0] for k in test_data.provide_data[0]]
    label_names = None if test_data.provide_label[0] is None else [k[0] for k in test_data.provide_label[0]]
    max_data_shape = [[('data', (1, 3, max([v[0] for v in cfg.SCALES]), max([v[1] for v in cfg.SCALES])))]]

    # create predictor
    predictor = Predictor(sym, data_names, label_names,
                          context=ctx, max_data_shapes=max_data_shape,
                          provide_data=test_data.provide_data, provide_label=test_data.provide_label,
                          arg_params=arg_params, aux_params=aux_params)

    # start testing
    imdb_boxes = generate_proposals(predictor, test_data, imdb, cfg, vis=vis, thresh=thresh)

    all_log_info = imdb.evaluate_recall(roidb, candidate_boxes=imdb_boxes)
    logger.info(all_log_info)
