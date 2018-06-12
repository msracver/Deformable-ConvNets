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

import mxnet as mx
import numpy as np

class FCNLogLossMetric(mx.metric.EvalMetric):
    def __init__(self, show_interval):
        super(FCNLogLossMetric, self).__init__('FCNLogLoss')
        self.show_interval = show_interval
        self.sum_metric = 0
        self.num_inst = 0

    def update(self, labels, preds):
        pred = preds[0]
        label = labels[0]

        # label (b, p)
        label = label.asnumpy().astype('int32').reshape((-1))
        # pred (b, c, p) or (b, c, h, w) --> (b, p, c) --> (b*p, c)
        pred = pred.asnumpy().reshape((pred.shape[0], pred.shape[1], -1)).transpose((0, 2, 1))
        pred = pred.reshape((label.shape[0], -1))

        # filter with keep_inds
        keep_inds = np.where(label != 255)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)

        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]
