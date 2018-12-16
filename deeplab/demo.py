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
import logging
import pprint
import cv2
from config.config import config, update_config
from utils.image import resize, transform
from PIL import Image
import numpy as np

# get config
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
cur_path = os.path.abspath(os.path.dirname(__file__))
update_config(cur_path + '/../experiments/deeplab/cfgs/deeplab_cityscapes_demo.yaml')

sys.path.insert(0, os.path.join(cur_path, '../external/mxnet', config.MXNET_VERSION))
import mxnet as mx
from core.tester import pred_eval, Predictor
from symbols import *
from utils.load_model import load_param
from utils.tictoc import tic, toc

def parse_args():
    parser = argparse.ArgumentParser(description='Show Deformable ConvNets demo')
    # general
    parser.add_argument('--deeplab_only', help='whether use Deeplab only (w/o Deformable ConvNets)', default=False, action='store_true')

    args = parser.parse_args()
    return args

args = parse_args()

def getpallete(num_cls):
    """
    this function is to get the colormap for visualizing the segmentation mask
    :param num_cls: the number of visulized class
    :return: the pallete
    """
    n = num_cls
    pallete_raw = np.zeros((n, 3)).astype('uint8')
    pallete = np.zeros((n, 3)).astype('uint8')

    pallete_raw[6, :] =  [111,  74,   0]
    pallete_raw[7, :] =  [ 81,   0,  81]
    pallete_raw[8, :] =  [128,  64, 128]
    pallete_raw[9, :] =  [244,  35, 232]
    pallete_raw[10, :] =  [250, 170, 160]
    pallete_raw[11, :] = [230, 150, 140]
    pallete_raw[12, :] = [ 70,  70,  70]
    pallete_raw[13, :] = [102, 102, 156]
    pallete_raw[14, :] = [190, 153, 153]
    pallete_raw[15, :] = [180, 165, 180]
    pallete_raw[16, :] = [150, 100, 100]
    pallete_raw[17, :] = [150, 120,  90]
    pallete_raw[18, :] = [153, 153, 153]
    pallete_raw[19, :] = [153, 153, 153]
    pallete_raw[20, :] = [250, 170,  30]
    pallete_raw[21, :] = [220, 220,   0]
    pallete_raw[22, :] = [107, 142,  35]
    pallete_raw[23, :] = [152, 251, 152]
    pallete_raw[24, :] = [ 70, 130, 180]
    pallete_raw[25, :] = [220,  20,  60]
    pallete_raw[26, :] = [255,   0,   0]
    pallete_raw[27, :] = [  0,   0, 142]
    pallete_raw[28, :] = [  0,   0,  70]
    pallete_raw[29, :] = [  0,  60, 100]
    pallete_raw[30, :] = [  0,   0,  90]
    pallete_raw[31, :] = [  0,   0, 110]
    pallete_raw[32, :] = [  0,  80, 100]
    pallete_raw[33, :] = [  0,   0, 230]
    pallete_raw[34, :] = [119,  11,  32]

    train2regular = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

    for i in range(len(train2regular)):
        pallete[i, :] = pallete_raw[train2regular[i]+1, :]

    pallete = pallete.reshape(-1)

    return pallete

def main():
    # get symbol
    pprint.pprint(config)
    config.symbol = 'resnet_v1_101_deeplab_dcn' if not args.deeplab_only else 'resnet_v1_101_deeplab'
    sym_instance = eval(config.symbol + '.' + config.symbol)()
    sym = sym_instance.get_symbol(config, is_train=False)

    # set up class names
    num_classes = 19

    # load demo data
    image_names = ['frankfurt_000001_073088_leftImg8bit.png', 'lindau_000024_000019_leftImg8bit.png']
    data = []
    for im_name in image_names:
        assert os.path.exists(cur_path + '/../demo/' + im_name), ('%s does not exist'.format('../demo/' + im_name))
        im = cv2.imread(cur_path + '/../demo/' + im_name, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        target_size = config.SCALES[0][0]
        max_size = config.SCALES[0][1]
        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
        data.append({'data': im_tensor, 'im_info': im_info})


    # get predictor
    data_names = ['data']
    label_names = ['softmax_label']
    data = [[mx.nd.array(data[i][name]) for name in data_names] for i in xrange(len(data))]
    max_data_shape = [[('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]]
    provide_data = [[(k, v.shape) for k, v in zip(data_names, data[i])] for i in xrange(len(data))]
    provide_label = [None for i in xrange(len(data))]
    arg_params, aux_params = load_param(cur_path + '/../model/' + ('deeplab_dcn_cityscapes' if not args.deeplab_only else 'deeplab_cityscapes'), 0, process=True)
    predictor = Predictor(sym, data_names, label_names,
                          context=[mx.gpu(0)], max_data_shapes=max_data_shape,
                          provide_data=provide_data, provide_label=provide_label,
                          arg_params=arg_params, aux_params=aux_params)

    # warm up
    for j in xrange(2):
        data_batch = mx.io.DataBatch(data=[data[0]], label=[], pad=0, index=0,
                                     provide_data=[[(k, v.shape) for k, v in zip(data_names, data[0])]],
                                     provide_label=[None])
        output_all = predictor.predict(data_batch)
        output_all = [mx.ndarray.argmax(output['softmax_output'], axis=1).asnumpy() for output in output_all]

    # test
    for idx, im_name in enumerate(image_names):
        data_batch = mx.io.DataBatch(data=[data[idx]], label=[], pad=0, index=idx,
                                     provide_data=[[(k, v.shape) for k, v in zip(data_names, data[idx])]],
                                     provide_label=[None])

        tic()
        output_all = predictor.predict(data_batch)
        output_all = [mx.ndarray.argmax(output['softmax_output'], axis=1).asnumpy() for output in output_all]
        pallete = getpallete(256)

        segmentation_result = np.uint8(np.squeeze(output_all))
        segmentation_result = Image.fromarray(segmentation_result)
        segmentation_result.putpalette(pallete)
        print 'testing {} {:.4f}s'.format(im_name, toc())
        pure_im_name, ext_im_name = os.path.splitext(im_name)
        segmentation_result.save(cur_path + '/../demo/seg_' + pure_im_name + '.png')
        # visualize
        im_raw = cv2.imread(cur_path + '/../demo/' + im_name)
        seg_res = cv2.imread(cur_path + '/../demo/seg_' + pure_im_name + '.png')
        cv2.imshow('Raw Image', im_raw)
        cv2.imshow('segmentation_result', seg_res)
        cv2.waitKey(0)
    print 'done'

if __name__ == '__main__':
    main()
