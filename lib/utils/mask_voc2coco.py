# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Yi Li
# --------------------------------------------------------

from skimage.draw import polygon
import numpy as np
import cv2
from utils.tictoc import tic, toc
from dataset.pycocotools.mask import encode as encodeMask_c

def encodeMask(M):
    """
    Encode binary mask M using run-length encoding.
    :param   M (bool 2D array)  : binary mask to encode
    :return: R (object RLE)     : run-length encoding of binary mask
    """
    [h, w] = M.shape
    M = M.flatten(order='F')
    N = len(M)
    counts_list = []
    pos = 0
    # counts
    counts_list.append(1)
    diffs = np.logical_xor(M[0:N - 1], M[1:N])
    for diff in diffs:
        if diff:
            pos += 1
            counts_list.append(1)
        else:
            counts_list[pos] += 1
    # if array starts from 1. start with 0 counts for 0
    if M[0] == 1:
        counts_list = [0] + counts_list
    return {'size': [h, w],
            'counts': counts_list,
            }

def mask_voc2coco(voc_masks, voc_boxes, im_height, im_width, binary_thresh = 0.4):
    num_pred = len(voc_masks)
    assert(num_pred==voc_boxes.shape[0])
    mask_img = np.zeros((im_height, im_width, num_pred), dtype=np.uint8, order='F')
    for i in xrange(num_pred):
        pred_box = np.round(voc_boxes[i, :4]).astype(int)
        pred_mask = voc_masks[i]
        pred_mask = cv2.resize(pred_mask.astype(np.float32), (pred_box[2] - pred_box[0] + 1, pred_box[3] - pred_box[1] + 1))
        mask_img[pred_box[1]:pred_box[3]+1, pred_box[0]:pred_box[2]+1, i] = pred_mask >= binary_thresh
    coco_mask = encodeMask_c(mask_img)
    return coco_mask
