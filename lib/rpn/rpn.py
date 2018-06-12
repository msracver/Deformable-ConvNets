# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Haozhi Qi
# --------------------------------------------------------
# Based on:
# MX-RCNN
# Copyright (c) 2016 by Contributors
# Licence under The Apache 2.0 License
# https://github.com/ijkguo/mx-rcnn/
# --------------------------------------------------------
"""
RPN:
data =
    {'data': [num_images, c, h, w],
     'im_info': [num_images, 4] (optional)}
label =
    {'gt_boxes': [num_boxes, 5] (optional),
     'label': [batch_size, 1] <- [batch_size, num_anchors, feat_height, feat_width],
     'bbox_target': [batch_size, num_anchors, feat_height, feat_width],
     'bbox_weight': [batch_size, num_anchors, feat_height, feat_width]}
"""

import numpy as np
import numpy.random as npr

from utils.image import get_image, tensor_vstack
from generate_anchor import generate_anchors
from bbox.bbox_transform import bbox_overlaps, bbox_transform


def get_rpn_testbatch(roidb, cfg):
    """
    return a dict of testbatch
    :param roidb: ['image', 'flipped']
    :return: data, label, im_info
    """
    # assert len(roidb) == 1, 'Single batch only'
    imgs, roidb = get_image(roidb, cfg)
    im_array = imgs
    im_info = [np.array([roidb[i]['im_info']], dtype=np.float32) for i in range(len(roidb))]

    data = [{'data': im_array[i],
            'im_info': im_info[i]} for i in range(len(roidb))]
    label = {}

    return data, label, im_info


def get_rpn_batch(roidb, cfg):
    """
    prototype for rpn batch: data, im_info, gt_boxes
    :param roidb: ['image', 'flipped'] + ['gt_boxes', 'boxes', 'gt_classes']
    :return: data, label
    """
    assert len(roidb) == 1, 'Single batch only'
    imgs, roidb = get_image(roidb, cfg)
    im_array = imgs[0]
    im_info = np.array([roidb[0]['im_info']], dtype=np.float32)

    # gt boxes: (x1, y1, x2, y2, cls)
    if roidb[0]['gt_classes'].size > 0:
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
        gt_boxes = np.empty((roidb[0]['boxes'].shape[0], 5), dtype=np.float32)
        gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :]
        gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
    else:
        gt_boxes = np.empty((0, 5), dtype=np.float32)

    data = {'data': im_array,
            'im_info': im_info}
    label = {'gt_boxes': gt_boxes}

    return data, label


def assign_anchor(feat_shape, gt_boxes, im_info, cfg, feat_stride=16,
                  scales=(8, 16, 32), ratios=(0.5, 1, 2), allowed_border=0):
    """
    assign ground truth boxes to anchor positions
    :param feat_shape: infer output shape
    :param gt_boxes: assign ground truth
    :param im_info: filter out anchors overlapped with edges
    :param feat_stride: anchor position step
    :param scales: used to generate anchors, affects num_anchors (per location)
    :param ratios: aspect ratios of generated anchors
    :param allowed_border: filter out anchors with edge overlap > allowed_border
    :return: dict of label
    'label': of shape (batch_size, 1) <- (batch_size, num_anchors, feat_height, feat_width)
    'bbox_target': of shape (batch_size, num_anchors * 4, feat_height, feat_width)
    'bbox_inside_weight': *todo* mark the assigned anchors
    'bbox_outside_weight': used to normalize the bbox_loss, all weights sums to RPN_POSITIVE_WEIGHT
    """
    def _unmap(data, count, inds, fill=0):
        """" unmap a subset inds of data into original data of size count """
        if len(data.shape) == 1:
            ret = np.empty((count,), dtype=np.float32)
            ret.fill(fill)
            ret[inds] = data
        else:
            ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
            ret.fill(fill)
            ret[inds, :] = data
        return ret

    DEBUG = False
    im_info = im_info[0]
    scales = np.array(scales, dtype=np.float32)
    base_anchors = generate_anchors(base_size=feat_stride, ratios=list(ratios), scales=scales)
    num_anchors = base_anchors.shape[0]
    feat_height, feat_width = feat_shape[-2:]

    if DEBUG:
        print 'anchors:'
        print base_anchors
        print 'anchor shapes:'
        print np.hstack((base_anchors[:, 2::4] - base_anchors[:, 0::4],
                         base_anchors[:, 3::4] - base_anchors[:, 1::4]))
        print 'im_info', im_info
        print 'height', feat_height, 'width', feat_width
        print 'gt_boxes shape', gt_boxes.shape
        print 'gt_boxes', gt_boxes

    # 1. generate proposals from bbox deltas and shifted anchors
    shift_x = np.arange(0, feat_width) * feat_stride
    shift_y = np.arange(0, feat_height) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = num_anchors
    K = shifts.shape[0]
    all_anchors = base_anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    all_anchors = all_anchors.reshape((K * A, 4))
    total_anchors = int(K * A)

    # only keep anchors inside the image
    inds_inside = np.where((all_anchors[:, 0] >= -allowed_border) &
                           (all_anchors[:, 1] >= -allowed_border) &
                           (all_anchors[:, 2] < im_info[1] + allowed_border) &
                           (all_anchors[:, 3] < im_info[0] + allowed_border))[0]
    if DEBUG:
        print 'total_anchors', total_anchors
        print 'inds_inside', len(inds_inside)

    # keep only inside anchors
    anchors = all_anchors[inds_inside, :]
    if DEBUG:
        print 'anchors shape', anchors.shape

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)

    if gt_boxes.size > 0:
        # overlap between the anchors and the gt boxes
        # overlaps (ex, gt)
        overlaps = bbox_overlaps(anchors.astype(np.float), gt_boxes.astype(np.float))
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels first so that positive labels can clobber them
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps] = 1

        # fg label: above threshold IoU
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels last so that negative labels can clobber positives
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
    else:
        labels[:] = 0

    # subsample positive labels if we have too many
    num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCH_SIZE)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        if DEBUG:
            disable_inds = fg_inds[:(len(fg_inds) - num_fg)]
        labels[disable_inds] = -1

    # subsample negative labels if we have too many
    num_bg = cfg.TRAIN.RPN_BATCH_SIZE - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        if DEBUG:
            disable_inds = bg_inds[:(len(bg_inds) - num_bg)]
        labels[disable_inds] = -1

    bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
    if gt_boxes.size > 0:
        bbox_targets[:] = bbox_transform(anchors, gt_boxes[argmax_overlaps, :4])

    bbox_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_WEIGHTS)

    if DEBUG:
        _sums = bbox_targets[labels == 1, :].sum(axis=0)
        _squared_sums = (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
        _counts = np.sum(labels == 1)
        means = _sums / (_counts + 1e-14)
        stds = np.sqrt(_squared_sums / _counts - means ** 2)
        print 'means', means
        print 'stdevs', stds

    # map up to original set of anchors
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_weights = _unmap(bbox_weights, total_anchors, inds_inside, fill=0)

    if DEBUG:
        print 'rpn: max max_overlaps', np.max(max_overlaps)
        print 'rpn: num_positives', np.sum(labels == 1)
        print 'rpn: num_negatives', np.sum(labels == 0)
        _fg_sum = np.sum(labels == 1)
        _bg_sum = np.sum(labels == 0)
        _count = 1
        print 'rpn: num_positive avg', _fg_sum / _count
        print 'rpn: num_negative avg', _bg_sum / _count

    labels = labels.reshape((1, feat_height, feat_width, A)).transpose(0, 3, 1, 2)
    labels = labels.reshape((1, A * feat_height * feat_width))
    bbox_targets = bbox_targets.reshape((1, feat_height, feat_width, A * 4)).transpose(0, 3, 1, 2)
    bbox_weights = bbox_weights.reshape((1, feat_height, feat_width, A * 4)).transpose((0, 3, 1, 2))

    label = {'label': labels,
             'bbox_target': bbox_targets,
             'bbox_weight': bbox_weights}
    return label


def assign_pyramid_anchor(feat_shapes, gt_boxes, im_info, cfg, feat_strides=(4, 8, 16, 32, 64),
                          scales=(8,), ratios=(0.5, 1, 2), allowed_border=0, balance_scale_bg=False,):
    """
    assign ground truth boxes to anchor positions
    :param feat_shapes: infer output shape
    :param gt_boxes: assign ground truth
    :param im_info: filter out anchors overlapped with edges
    :param feat_strides: anchor position step
    :param scales: used to generate anchors, affects num_anchors (per location)
    :param ratios: aspect ratios of generated anchors
    :param allowed_border: filter out anchors with edge overlap > allowed_border
    :param balance_scale_bg: restrict the background samples for each pyramid level
    :return: dict of label
    'label': of shape (batch_size, 1) <- (batch_size, num_anchors, feat_height, feat_width)
    'bbox_target': of shape (batch_size, num_anchors * 4, feat_height, feat_width)
    'bbox_inside_weight': *todo* mark the assigned anchors
    'bbox_outside_weight': used to normalize the bbox_loss, all weights sums to RPN_POSITIVE_WEIGHT
    """
    def _unmap(data, count, inds, fill=0):
        """" unmap a subset inds of data into original data of size count """
        if len(data.shape) == 1:
            ret = np.empty((count,), dtype=np.float32)
            ret.fill(fill)
            ret[inds] = data
        else:
            ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
            ret.fill(fill)
            ret[inds, :] = data
        return ret

    DEBUG = False
    im_info = im_info[0]
    scales = np.array(scales, dtype=np.float32)
    ratios = np.array(ratios, dtype=np.float32)
    assert(len(feat_shapes) == len(feat_strides))

    fpn_args = []
    fpn_anchors_fid = np.zeros(0).astype(int)
    fpn_anchors = np.zeros([0, 4])
    fpn_labels = np.zeros(0)
    fpn_inds_inside = []
    for feat_id in range(len(feat_strides)):
        # len(scales.shape) == 1 just for backward compatibility, will remove in the future
        if len(scales.shape) == 1:
            base_anchors = generate_anchors(base_size=feat_strides[feat_id], ratios=ratios, scales=scales)
        else:
            assert len(scales.shape) == len(ratios.shape) == 2
            base_anchors = generate_anchors(base_size=feat_strides[feat_id], ratios=ratios[feat_id], scales=scales[feat_id])
        num_anchors = base_anchors.shape[0]
        feat_height, feat_width = feat_shapes[feat_id][0][-2:]

        # 1. generate proposals from bbox deltas and shifted anchors
        shift_x = np.arange(0, feat_width) * feat_strides[feat_id]
        shift_y = np.arange(0, feat_height) * feat_strides[feat_id]
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = num_anchors
        K = shifts.shape[0]
        all_anchors = base_anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        all_anchors = all_anchors.reshape((K * A, 4))
        total_anchors = int(K * A)

        # only keep anchors inside the image
        inds_inside = np.where((all_anchors[:, 0] >= -allowed_border) &
                               (all_anchors[:, 1] >= -allowed_border) &
                               (all_anchors[:, 2] < im_info[1] + allowed_border) &
                               (all_anchors[:, 3] < im_info[0] + allowed_border))[0]

        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]

        # label: 1 is positive, 0 is negative, -1 is dont care
        # for sigmoid classifier, ignore the 'background' class
        labels = np.empty((len(inds_inside),), dtype=np.float32)
        labels.fill(-1)

        fpn_anchors_fid = np.hstack((fpn_anchors_fid, len(inds_inside)))
        fpn_anchors = np.vstack((fpn_anchors, anchors))
        fpn_labels = np.hstack((fpn_labels, labels))
        fpn_inds_inside.append(inds_inside)
        fpn_args.append([feat_height, feat_width, A, total_anchors])

    if gt_boxes.size > 0:
        # overlap between the anchors and the gt boxes
        # overlaps (ex, gt)
        overlaps = bbox_overlaps(fpn_anchors.astype(np.float), gt_boxes.astype(np.float))
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(fpn_anchors)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels first so that positive labels can clobber them
            fpn_labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
        # fg label: for each gt, anchor with highest overlap
        fpn_labels[gt_argmax_overlaps] = 1
        # fg label: above threshold IoU
        fpn_labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1
        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels last so that negative labels can clobber positives
            fpn_labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
    else:
        fpn_labels[:] = 0

    # subsample positive labels if we have too many
    num_fg = fpn_labels.shape[0] if cfg.TRAIN.RPN_BATCH_SIZE == -1 else int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCH_SIZE)
    fg_inds = np.where(fpn_labels >= 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        if DEBUG:
            disable_inds = fg_inds[:(len(fg_inds) - num_fg)]
        fpn_labels[disable_inds] = -1

    # subsample negative labels if we have too many
    num_bg = fpn_labels.shape[0] if cfg.TRAIN.RPN_BATCH_SIZE == -1 else cfg.TRAIN.RPN_BATCH_SIZE - np.sum(fpn_labels >= 1)
    bg_inds = np.where(fpn_labels == 0)[0]
    fpn_anchors_fid = np.hstack((0, fpn_anchors_fid.cumsum()))

    if balance_scale_bg:
        num_bg_scale = num_bg / len(feat_strides)
        for feat_id in range(0, len(feat_strides)):
            bg_ind_scale = bg_inds[(bg_inds >= fpn_anchors_fid[feat_id]) & (bg_inds < fpn_anchors_fid[feat_id+1])]
            if len(bg_ind_scale) > num_bg_scale:
                disable_inds = npr.choice(bg_ind_scale, size=(len(bg_ind_scale) - num_bg_scale), replace=False)
                fpn_labels[disable_inds] = -1
    else:
        if len(bg_inds) > num_bg:
            disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            if DEBUG:
                disable_inds = bg_inds[:(len(bg_inds) - num_bg)]
            fpn_labels[disable_inds] = -1

    fpn_bbox_targets = np.zeros((len(fpn_anchors), 4), dtype=np.float32)
    if gt_boxes.size > 0:
        fpn_bbox_targets[fpn_labels >= 1, :] = bbox_transform(fpn_anchors[fpn_labels >= 1, :], gt_boxes[argmax_overlaps[fpn_labels >= 1], :4])
        # fpn_bbox_targets[:] = bbox_transform(fpn_anchors, gt_boxes[argmax_overlaps, :4])
    # fpn_bbox_targets = (fpn_bbox_targets - np.array(cfg.TRAIN.BBOX_MEANS)) / np.array(cfg.TRAIN.BBOX_STDS)
    fpn_bbox_weights = np.zeros((len(fpn_anchors), 4), dtype=np.float32)

    fpn_bbox_weights[fpn_labels >= 1, :] = np.array(cfg.TRAIN.RPN_BBOX_WEIGHTS)

    label_list = []
    bbox_target_list = []
    bbox_weight_list = []
    for feat_id in range(0, len(feat_strides)):
        feat_height, feat_width, A, total_anchors = fpn_args[feat_id]
        # map up to original set of anchors
        labels = _unmap(fpn_labels[fpn_anchors_fid[feat_id]:fpn_anchors_fid[feat_id+1]], total_anchors, fpn_inds_inside[feat_id], fill=-1)
        bbox_targets = _unmap(fpn_bbox_targets[fpn_anchors_fid[feat_id]:fpn_anchors_fid[feat_id+1]], total_anchors, fpn_inds_inside[feat_id], fill=0)
        bbox_weights = _unmap(fpn_bbox_weights[fpn_anchors_fid[feat_id]:fpn_anchors_fid[feat_id+1]], total_anchors, fpn_inds_inside[feat_id], fill=0)

        labels = labels.reshape((1, feat_height, feat_width, A)).transpose(0, 3, 1, 2)
        labels = labels.reshape((1, A * feat_height * feat_width))

        bbox_targets = bbox_targets.reshape((1, feat_height, feat_width, A * 4)).transpose(0, 3, 1, 2)
        bbox_targets = bbox_targets.reshape((1, A * 4, -1))
        bbox_weights = bbox_weights.reshape((1, feat_height, feat_width, A * 4)).transpose((0, 3, 1, 2))
        bbox_weights = bbox_weights.reshape((1, A * 4, -1))

        label_list.append(labels)
        bbox_target_list.append(bbox_targets)
        bbox_weight_list.append(bbox_weights)
        # label.update({'label_p' + str(feat_id + feat_id_start): labels,
        #               'bbox_target_p' + str(feat_id + feat_id_start): bbox_targets,
        #               'bbox_weight_p' + str(feat_id + feat_id_start): bbox_weights})

    label = {
        'label': np.concatenate(label_list, axis=1),
        'bbox_target': np.concatenate(bbox_target_list, axis=2),
        'bbox_weight': np.concatenate(bbox_weight_list, axis=2)
    }

    return label
