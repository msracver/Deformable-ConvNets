# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuwen Xiong
# --------------------------------------------------------

"""
roidb
basic format [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
extended ['image', 'max_classes', 'max_overlaps', 'bbox_targets']
"""

import cv2
import numpy as np

from bbox.bbox_regression import compute_bbox_regression_targets


def prepare_roidb(imdb, roidb, cfg):
    """
    add image path, max_classes, max_overlaps to roidb
    :param imdb: image database, provide path
    :param roidb: roidb
    :return: None
    """
    print 'prepare roidb'
    for i in range(len(roidb)):  # image_index
        roidb[i]['image'] = imdb.image_path_from_index(imdb.image_set_index[i])
        if cfg.TRAIN.ASPECT_GROUPING:
            size = cv2.imread(roidb[i]['image']).shape
            roidb[i]['height'] = size[0]
            roidb[i]['width'] = size[1]
        gt_overlaps = roidb[i]['gt_overlaps'].toarray()
        max_overlaps = gt_overlaps.max(axis=1)
        max_classes = gt_overlaps.argmax(axis=1)
        roidb[i]['max_overlaps'] = max_overlaps
        roidb[i]['max_classes'] = max_classes

        # background roi => background class
        zero_indexes = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_indexes] == 0)
        # foreground roi => foreground class
        nonzero_indexes = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_indexes] != 0)
