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
update_config(cur_path + '/../experiments/rfcn/cfgs/deform_conv_demo.yaml')

sys.path.insert(0, os.path.join(cur_path, '../external/mxnet', config.MXNET_VERSION))
import mxnet as mx
from core.tester import Predictor
from symbols import *
from utils.load_model import load_param
from utils.show_offset import show_dconv_offset

def main():
    # get symbol
    pprint.pprint(config)
    sym_instance = eval(config.symbol + '.' + config.symbol)()
    sym = sym_instance.get_symbol(config, is_train=False)

    # load demo data
    image_names = ['000240.jpg', '000437.jpg', '004072.jpg', '007912.jpg']
    image_all = []
    data = []
    for im_name in image_names:
        assert os.path.exists(cur_path + '/../demo/deform_conv/' + im_name), \
            ('%s does not exist'.format('../demo/deform_conv/' + im_name))
        im = cv2.imread(cur_path + '/../demo/deform_conv/' + im_name, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        image_all.append(im)
        target_size = config.SCALES[0][0]
        max_size = config.SCALES[0][1]
        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
        data.append({'data': im_tensor, 'im_info': im_info})

    # get predictor
    data_names = ['data', 'im_info']
    label_names = []
    data = [[mx.nd.array(data[i][name]) for name in data_names] for i in xrange(len(data))]
    max_data_shape = [[('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]]
    provide_data = [[(k, v.shape) for k, v in zip(data_names, data[i])] for i in xrange(len(data))]
    provide_label = [None for i in xrange(len(data))]
    arg_params, aux_params = load_param(cur_path + '/../model/deform_conv', 0, process=True)
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
        res5a_offset = output[0]['res5a_branch2b_offset_output'].asnumpy()
        res5b_offset = output[0]['res5b_branch2b_offset_output'].asnumpy()
        res5c_offset = output[0]['res5c_branch2b_offset_output'].asnumpy()

        im = image_all[idx]
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        show_dconv_offset(im, [res5c_offset, res5b_offset, res5a_offset])

if __name__ == '__main__':
    main()
