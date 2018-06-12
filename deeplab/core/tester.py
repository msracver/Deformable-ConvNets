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

import cPickle
import os
import time
import mxnet as mx
import numpy as np

from PIL import Image
from module import MutableModule
from config.config import config
from utils import image
from utils.PrefetchingIter import PrefetchingIter


class Predictor(object):
    def __init__(self, symbol, data_names, label_names,
                 context=mx.cpu(), max_data_shapes=None,
                 provide_data=None, provide_label=None,
                 arg_params=None, aux_params=None):
        self._mod = MutableModule(symbol, data_names, label_names,
                                  context=context, max_data_shapes=max_data_shapes)
        self._mod.bind(provide_data, provide_label, for_training=False)
        self._mod.init_params(arg_params=arg_params, aux_params=aux_params)

    def predict(self, data_batch):
        self._mod.forward(data_batch)
        # [dict(zip(self._mod.output_names, _)) for _ in zip(*self._mod.get_outputs(merge_multi_context=False))]
        return [dict(zip(self._mod.output_names, _)) for _ in zip(*self._mod.get_outputs(merge_multi_context=False))]

def pred_eval(predictor, test_data, imdb, vis=False, ignore_cache=None, logger=None):
    """
    wrapper for calculating offline validation for faster data analysis
    in this example, all threshold are set by hand
    :param predictor: Predictor
    :param test_data: data iterator, must be non-shuffle
    :param imdb: image database
    :param vis: controls visualization
    :param ignore_cache: ignore the saved cache file
    :param logger: the logger instance
    :return:
    """
    res_file = os.path.join(imdb.result_path, imdb.name + '_segmentations.pkl')
    if os.path.exists(res_file) and not ignore_cache:
        with open(res_file , 'rb') as fid:
            evaluation_results = cPickle.load(fid)
        print 'evaluate segmentation: \n'
        if logger:
            logger.info('evaluate segmentation: \n')

        meanIU = evaluation_results['meanIU']
        IU_array = evaluation_results['IU_array']
        print 'IU_array:\n'
        if logger:
            logger.info('IU_array:\n')
        for i in range(len(IU_array)):
            print '%.5f'%IU_array[i]
            if logger:
                logger.info('%.5f'%IU_array[i])
        print 'meanIU:%.5f'%meanIU
        if logger:
            logger.info( 'meanIU:%.5f'%meanIU)
        return

    assert vis or not test_data.shuffle
    if not isinstance(test_data, PrefetchingIter):
        test_data = PrefetchingIter(test_data)

    num_images = imdb.num_images
    all_segmentation_result = [[] for _ in xrange(num_images)]
    idx = 0

    data_time, net_time, post_time = 0.0, 0.0, 0.0
    t = time.time()
    for data_batch in test_data:
        t1 = time.time() - t
        t = time.time()
        output_all = predictor.predict(data_batch)
        output_all = [mx.ndarray.argmax(output['softmax_output'], axis=1).asnumpy() for output in output_all]
        t2 = time.time() - t
        t = time.time()

        all_segmentation_result[idx: idx+test_data.batch_size] = [output.astype('int8') for output in output_all]

        idx += test_data.batch_size
        t3 = time.time() - t
        t = time.time()

        data_time += t1
        net_time += t2
        post_time += t3
        print 'testing {}/{} data {:.4f}s net {:.4f}s post {:.4f}s'.format(idx, imdb.num_images, data_time / idx * test_data.batch_size, net_time / idx * test_data.batch_size, post_time / idx * test_data.batch_size)
        if logger:
            logger.info('testing {}/{} data {:.4f}s net {:.4f}s post {:.4f}s'.format(idx, imdb.num_images, data_time / idx * test_data.batch_size, net_time / idx * test_data.batch_size, post_time / idx * test_data.batch_size))

    evaluation_results = imdb.evaluate_segmentations(all_segmentation_result)

    if not os.path.exists(res_file) or ignore_cache:
        with open(res_file, 'wb') as f:
            cPickle.dump(evaluation_results, f, protocol=cPickle.HIGHEST_PROTOCOL)

    print 'evaluate segmentation: \n'
    if logger:
        logger.info('evaluate segmentation: \n')

    meanIU = evaluation_results['meanIU']
    IU_array = evaluation_results['IU_array']
    print 'IU_array:\n'
    if logger:
        logger.info('IU_array:\n')
    for i in range(len(IU_array)):
        print '%.5f'%IU_array[i]
        if logger:
            logger.info('%.5f'%IU_array[i])
    print 'meanIU:%.5f'%meanIU
    if logger:
        logger.info( 'meanIU:%.5f'%meanIU)
