# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Haozhi Qi, Yuwen Xiong
# --------------------------------------------------------

import mxnet as mx
import numpy as np
from mxnet.contrib import autograd
import gc


class FPNROIPoolingOperator(mx.operator.CustomOp):
    def __init__(self, feat_strides, pooled_height, pooled_width, output_dim, with_deformable):
        self.pooled_height = pooled_height
        self.pooled_width = pooled_width
        self.feat_strides = feat_strides
        self.with_deformable = with_deformable
        self.output_dim = output_dim
        self.in_grad_hist_list = []
        self.num_strides = len(self.feat_strides)
        self.roi_pool = [None for _ in range(self.num_strides)]
        self.feat_idx = [None for _ in range(self.num_strides)]

    def forward(self, is_train, req, in_data, out_data, aux):
        rois = in_data[-1].asnumpy()
        w = rois[:, 3] - rois[:, 1] + 1
        h = rois[:, 4] - rois[:, 2] + 1
        feat_id = np.clip(np.floor(2 + np.log2(np.sqrt(w * h) / 224)), 0, len(self.feat_strides) - 1)
        pyramid_idx = []

        rois_p = [None for _ in range(self.num_strides)]
        for i in range(self.num_strides):
            self.feat_idx[i] = np.where(feat_id == i)[0]
            if len(self.feat_idx[i]) == 0:
                # padding dummy roi
                rois_p[i] = np.zeros((1, 5))
                pyramid_idx.append(-1)
            else:
                rois_p[i] = rois[self.feat_idx[i]]
                pyramid_idx.append(self.feat_idx[i])
        rois_idx = np.argsort(np.hstack(pyramid_idx))[-rois.shape[0]:]

        if is_train:
            for i in range(self.num_strides):
                self.in_grad_hist_list.append(mx.nd.zeros_like(in_data[i]))

            if self.with_deformable:
                for i in range(self.num_strides, self.num_strides * 3):
                    self.in_grad_hist_list.append(mx.nd.zeros_like(in_data[i]))
                autograd.mark_variables([in_data[i] for i in range(self.num_strides * 3)], self.in_grad_hist_list)

                with autograd.train_section():
                    for i in range(self.num_strides):
                        roi_offset_t = mx.contrib.nd.DeformablePSROIPooling(data=in_data[i], rois=mx.nd.array(rois_p[i], in_data[i].context), group_size=1, pooled_size=7,
                                                                            sample_per_part=4, no_trans=True, part_size=7, output_dim=256, spatial_scale=1.0 / self.feat_strides[i])
                        roi_offset = mx.nd.FullyConnected(data=roi_offset_t, num_hidden=7 * 7 * 2, weight=in_data[i * 2 + self.num_strides], bias=in_data[i * 2 + 1 + self.num_strides])
                        roi_offset_reshape = mx.nd.reshape(data=roi_offset, shape=(-1, 2, 7, 7))
                        self.roi_pool[i] = mx.contrib.nd.DeformablePSROIPooling(data=in_data[i], rois=mx.nd.array(rois_p[i], in_data[i].context), trans=roi_offset_reshape,
                                                                                group_size=1, pooled_size=7, sample_per_part=4, no_trans=False, part_size=7,
                                                                                output_dim=self.output_dim, spatial_scale=1.0 / self.feat_strides[i], trans_std=0.1)
            else:
                autograd.mark_variables([in_data[i] for i in range(self.num_strides)], self.in_grad_hist_list)
                with autograd.train_section():
                    for i in range(self.num_strides):
                        self.roi_pool[i] = mx.nd.ROIPooling(in_data[i], mx.nd.array(rois_p[i], in_data[i].context), (7, 7), spatial_scale=1.0 / self.feat_strides[i])
            roi_pool = mx.nd.concatenate(self.roi_pool, axis=0)
        else:
            # during testing, there is no need to record variable, thus saving memory
            roi_pool = [None for _ in range(self.num_strides)]
            if self.with_deformable:
                for i in range(self.num_strides):
                    roi_offset_t = mx.contrib.nd.DeformablePSROIPooling(data=in_data[i], rois=mx.nd.array(rois_p[i], in_data[i].context), group_size=1, pooled_size=7,
                                                                        sample_per_part=4, no_trans=True, part_size=7, output_dim=256, spatial_scale=1.0 / self.feat_strides[i])
                    roi_offset = mx.nd.FullyConnected(data=roi_offset_t, num_hidden=7 * 7 * 2, weight=in_data[i * 2 + self.num_strides], bias=in_data[i * 2 + 1 + self.num_strides])
                    roi_offset_reshape = mx.nd.reshape(data=roi_offset, shape=(-1, 2, 7, 7))
                    roi_pool[i] = mx.contrib.nd.DeformablePSROIPooling(data=in_data[i], rois=mx.nd.array(rois_p[i], in_data[i].context), trans=roi_offset_reshape,
                                                                       group_size=1, pooled_size=7, sample_per_part=4, no_trans=False, part_size=7,
                                                                       output_dim=self.output_dim, spatial_scale=1.0 / self.feat_strides[i], trans_std=0.1)
            else:
                for i in range(self.num_strides):
                    roi_pool[i] = mx.nd.ROIPooling(in_data[i], mx.nd.array(rois_p[i], in_data[i].context), (7, 7), spatial_scale=1.0 / self.feat_strides[i])

            roi_pool = mx.nd.concatenate(roi_pool, axis=0)

        roi_pool = mx.nd.take(roi_pool, mx.nd.array(rois_idx, roi_pool.context))
        self.assign(out_data[0], req[0], roi_pool)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for i in range(len(in_grad)):
            self.assign(in_grad[i], req[i], 0)

        with autograd.train_section():
            for i in range(self.num_strides):
                if len(self.feat_idx[i] > 0):
                    autograd.compute_gradient([mx.nd.take(out_grad[0], mx.nd.array(self.feat_idx[i], out_grad[0].context)) * self.roi_pool[i]])

        if self.with_deformable:
            for i in range(0, self.num_strides * 3):
                self.assign(in_grad[i], req[i], self.in_grad_hist_list[i])
        else:
            for i in range(0, self.num_strides):
                self.assign(in_grad[i], req[i], self.in_grad_hist_list[i])

        gc.collect()


@mx.operator.register('fpn_roi_pooling')
class FPNROIPoolingProp(mx.operator.CustomOpProp):
    def __init__(self, feat_strides='(4,8,16,32)', pooled_height='7', pooled_width='7', with_deformable='False', output_dim='256'):
        super(FPNROIPoolingProp, self).__init__(need_top_grad=True)
        self.pooled_height = int(pooled_height)
        self.pooled_width = int(pooled_width)
        self.feat_strides = np.fromstring(feat_strides[1:-1], dtype=int, sep=',')
        self.with_deformable = with_deformable == 'True'
        self.output_dim = int(output_dim)

        self.num_strides = len(self.feat_strides)

    def list_arguments(self):
        args_list = []
        for i in range(self.num_strides):
            args_list.append('data_p{}'.format(2 + i))
        if self.with_deformable:
            for i in range(self.num_strides):
                args_list.extend(['offset_weight_p{}'.format(2 + i), 'offset_bias_p{}'.format(2 + i)])
        args_list.append('rois')
        return args_list

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        output_feat_shape = [in_shape[-1][0], in_shape[0][1], self.pooled_height, self.pooled_width]
        if self.with_deformable:
            offset_dim = self.pooled_height * self.pooled_width * 2
            input_dim = self.pooled_height * self.pooled_width * self.output_dim
            for i in range(self.num_strides):
                in_shape[i * 2 + self.num_strides], in_shape[i * 2 + 1 + self.num_strides] = [offset_dim, input_dim], [offset_dim, ]
        return in_shape, [output_feat_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return FPNROIPoolingOperator(self.feat_strides, self.pooled_height, self.pooled_width, self.output_dim, self.with_deformable)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return [out_grad[0]]
