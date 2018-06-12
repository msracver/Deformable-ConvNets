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

import numpy as np
import mxnet as mx
import random
import math

from mxnet.executor_manager import _split_input_slice
from utils.image import tensor_vstack
from segmentation.segmentation import get_segmentation_train_batch, get_segmentation_test_batch
from PIL import Image
from multiprocessing import Pool

class TestDataLoader(mx.io.DataIter):
    def __init__(self, segdb, config, batch_size=1, shuffle=False):
        super(TestDataLoader, self).__init__()

        # save parameters as properties
        self.segdb = segdb
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.config = config

        # infer properties from roidb
        self.size = len(self.segdb)
        self.index = np.arange(self.size)

        # decide data and label names (only for training)
        self.data_name = ['data']
        self.label_name = None

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.data = None
        self.label = []
        self.im_info = None

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.get_batch()

    @property
    def provide_data(self):
        return [[(k, v.shape) for k, v in zip(self.data_name, self.data[i])] for i in xrange(len(self.data))]

    @property
    def provide_label(self):
        return [None for i in xrange(len(self.data))]

    @property
    def provide_data_single(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data[0])]

    @property
    def provide_label_single(self):
        return None

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur < self.size

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        segdb = [self.segdb[self.index[i]] for i in range(cur_from, cur_to)]

        data, label, im_info = get_segmentation_test_batch(segdb, self.config)

        self.data = [[mx.nd.array(data[i][name]) for name in self.data_name] for i in xrange(len(data))]
        self.im_info = im_info

class TrainDataLoader(mx.io.DataIter):
    def __init__(self, sym, segdb, config, batch_size=1, crop_height = 768, crop_width = 1024, shuffle=False, ctx=None, work_load_list=None):
        """
        This Iter will provide seg data to Deeplab network
        :param sym: to infer shape
        :param segdb: must be preprocessed
        :param config: config file
        :param batch_size: must divide BATCH_SIZE(128)
        :param crop_height: the height of cropped image
        :param crop_width: the width of cropped image
        :param shuffle: bool
        :param ctx: list of contexts
        :param work_load_list: list of work load
        :return: DataLoader
        """
        super(TrainDataLoader, self).__init__()

        # save parameters as properties
        self.sym = sym
        self.segdb = segdb
        self.config = config
        self.batch_size = batch_size
        if self.config.TRAIN.ENABLE_CROP:
            self.crop_height = crop_height
            self.crop_width = crop_width
        else:
            self.crop_height = None
            self.crop_width = None

        self.shuffle = shuffle
        self.ctx = ctx

        if self.ctx is None:
            self.ctx = [mx.cpu()]
        self.work_load_list = work_load_list

        # infer properties from segdb
        self.size = len(segdb)
        self.index = np.arange(self.size)

        # decide data and label names
        self.data_name = ['data']
        self.label_name = ['label']

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.batch = None
        self.data = None
        self.label = None

        # init multi-process pool
        self.pool = Pool(processes = len(self.ctx))

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.get_batch_parallel()
        random.seed()

    @property
    def provide_data(self):
        return [[(k, v.shape) for k, v in zip(self.data_name, self.data[i])] for i in xrange(len(self.data))]

    @property
    def provide_label(self):
        return [[(k, v.shape) for k, v in zip(self.label_name, self.label[i])] for i in xrange(len(self.data))]

    @property
    def provide_data_single(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data[0])]

    @property
    def provide_label_single(self):
        return [(k, v.shape) for k, v in zip(self.label_name, self.label[0])]

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch_parallel()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def infer_shape(self, max_data_shape=None, max_label_shape=None):
        """ Return maximum data and label shape for single gpu """
        if max_data_shape is None:
            max_data_shape = []
        if max_label_shape is None:
            max_label_shape = []

        max_shapes = dict(max_data_shape + max_label_shape)
        _, label_shape, _ = self.sym.infer_shape(**max_shapes)
        label_shape = [(self.label_name[0], label_shape)]
        return max_data_shape, label_shape

    def get_batch_parallel(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        segdb = [self.segdb[self.index[i]] for i in range(cur_from, cur_to)]

        # decide multi device slice
        work_load_list = self.work_load_list
        ctx = self.ctx
        if work_load_list is None:
            work_load_list = [1] * len(ctx)
        assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
            "Invalid settings for work load. "
        slices = _split_input_slice(self.batch_size, work_load_list)

        multiprocess_results = []

        for idx, islice in enumerate(slices):
            isegdb = [segdb[i] for i in range(islice.start, islice.stop)]
            multiprocess_results.append(self.pool.apply_async(parfetch, (self.config, self.crop_width, self.crop_height, isegdb)))

        rst = [multiprocess_result.get() for multiprocess_result in multiprocess_results]

        all_data = [_['data'] for _ in rst]
        all_label = [_['label'] for _ in rst]
        self.data = [[mx.nd.array(data[key]) for key in self.data_name] for data in all_data]
        self.label = [[mx.nd.array(label[key]) for key in self.label_name] for label in all_label]

def parfetch(config, crop_width, crop_height, isegdb):
    # get testing data for multigpu
    data, label = get_segmentation_train_batch(isegdb, config)
    if config.TRAIN.ENABLE_CROP:
        data_internal = data['data']
        label_internal = label['label']

        sx = math.floor(random.random() * (data_internal.shape[3] - crop_width + 1))
        sy = math.floor(random.random() * (data_internal.shape[2] - crop_height + 1))
        sx = (int)(sx)
        sy = (int)(sy)
        assert(sx >= 0 and sx < data_internal.shape[3] - crop_width + 1)
        assert(sy >= 0 and sy < data_internal.shape[2] - crop_height + 1)

        ex = (int)(sx + crop_width - 1)
        ey = (int)(sy + crop_height - 1)

        data_internal = data_internal[:, :, sy : ey + 1, sx : ex + 1]
        label_internal = label_internal[:, :, sy : ey + 1, sx : ex + 1]

        data['data'] = data_internal
        label['label'] = label_internal
        assert (data['data'].shape[2] == crop_height) and (data['data'].shape[3] == crop_width)
        assert (label['label'].shape[2] == crop_height) and (label['label'].shape[3] == crop_width)

    return {'data': data, 'label': label}
