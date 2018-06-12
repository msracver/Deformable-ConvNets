# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Yi Li, Haochen Zhang
# --------------------------------------------------------

import _init_paths

import os
import sys
import pprint
import cv2
from config.config import config, update_config
from utils.image import resize, transform
import numpy as np
# get config
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
cur_path = os.path.abspath(os.path.dirname(__file__))
update_config(cur_path + '/../experiments/rfcn/cfgs/deform_psroi_demo.yaml')

sys.path.insert(0, os.path.join(cur_path, '../external/mxnet', config.MXNET_VERSION))
import mxnet as mx
from core.tester import Predictor
from symbols import *
from utils.load_model import load_param
from utils.show_offset import show_dpsroi_offset

def main():
    # get symbol
    pprint.pprint(config)
    sym_instance = eval(config.symbol + '.' + config.symbol)()
    sym = sym_instance.get_symbol_rfcn(config, is_train=False)

    # load demo data
    image_names = ['000057.jpg', '000149.jpg', '000351.jpg', '002535.jpg']
    image_all = []
    # ground truth boxes
    gt_boxes_all = [np.array([[132, 52, 384, 357]]), np.array([[113, 1, 350, 360]]),
                    np.array([[0, 27, 329, 155]]), np.array([[8, 40, 499, 289]])]
    gt_classes_all = [np.array([3]), np.array([16]), np.array([7]), np.array([12])]
    data = []
    for idx, im_name in enumerate(image_names):
        assert os.path.exists(cur_path + '/../demo/deform_psroi/' + im_name), \
            ('%s does not exist'.format('../demo/deform_psroi/' + im_name))
        im = cv2.imread(cur_path + '/../demo/deform_psroi/' + im_name, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        image_all.append(im)
        target_size = config.SCALES[0][0]
        max_size = config.SCALES[0][1]
        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        gt_boxes = gt_boxes_all[idx]
        gt_boxes = np.round(gt_boxes * im_scale)
        data.append({'data': im_tensor, 'rois': np.hstack((np.zeros((gt_boxes.shape[0], 1)), gt_boxes))})

    # get predictor
    data_names = ['data', 'rois']
    label_names = []
    data = [[mx.nd.array(data[i][name]) for name in data_names] for i in xrange(len(data))]
    max_data_shape = [[('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]]
    provide_data = [[(k, v.shape) for k, v in zip(data_names, data[i])] for i in xrange(len(data))]
    provide_label = [None for i in xrange(len(data))]
    arg_params, aux_params = load_param(cur_path + '/../model/deform_psroi', 0, process=True)
    predictor = Predictor(sym, data_names, label_names,
                          context=[mx.gpu(0)], max_data_shapes=max_data_shape,
                          provide_data=provide_data, provide_label=provide_label,
                          arg_params=arg_params, aux_params=aux_params)

    # test
    for idx, _ in enumerate(image_names):
        data_batch = mx.io.DataBatch(data=[data[idx]], label=[], pad=0, index=idx,
                                     provide_data=[[(k, v.shape) for k, v in zip(data_names, data[idx])]],
                                     provide_label=[None])

        output = predictor.predict(data_batch)
        cls_offset = output[0]['rfcn_cls_offset_output'].asnumpy()

        im = image_all[idx]
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        boxes = gt_boxes_all[idx]
        show_dpsroi_offset(im, boxes, cls_offset, gt_classes_all[idx])

if __name__ == '__main__':
    main()
