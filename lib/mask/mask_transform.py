# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Haozhi Qi, Yi Li, Guodong Zhang
# --------------------------------------------------------

import numpy as np


def intersect_box_mask(ex_box, gt_box, gt_mask):
    """
    This function calculate the intersection part of a external box
    and gt_box, mask it according to gt_mask
    Args:
        ex_box: external ROIS
        gt_box: ground truth boxes
        gt_mask: ground truth masks, not been resized yet
    Returns:
        regression_target: logical numpy array
    """
    x1 = max(ex_box[0], gt_box[0])
    y1 = max(ex_box[1], gt_box[1])
    x2 = min(ex_box[2], gt_box[2])
    y2 = min(ex_box[3], gt_box[3])
    if x1 > x2 or y1 > y2:
        return np.zeros((21, 21), dtype=bool)
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    ex_starty = y1 - ex_box[1]
    ex_startx = x1 - ex_box[0]

    inter_maskb = gt_mask[y1:y2+1 , x1:x2+1]
    regression_target = np.zeros((ex_box[3] - ex_box[1] + 1, ex_box[2] - ex_box[0] + 1))
    regression_target[ex_starty: ex_starty + h, ex_startx: ex_startx + w] = inter_maskb

    return regression_target


def mask_overlap(box1, box2, mask1, mask2):
    """
    This function calculate region IOU when masks are
    inside different boxes
    Returns:
        intersection over unions of this two masks
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x1 > x2 or y1 > y2:
        return 0
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    # get masks in the intersection part
    start_ya = y1 - box1[1]
    start_xa = x1 - box1[0]
    inter_maska = mask1[start_ya: start_ya + h, start_xa:start_xa + w]

    start_yb = y1 - box2[1]
    start_xb = x1 - box2[0]
    inter_maskb = mask2[start_yb: start_yb + h, start_xb:start_xb + w]

    assert inter_maska.shape == inter_maskb.shape

    inter = np.logical_and(inter_maskb, inter_maska).sum()
    union = mask1.sum() + mask2.sum() - inter
    if union < 1.0:
        return 0
    return float(inter) / float(union)
