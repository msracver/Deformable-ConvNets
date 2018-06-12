# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Yi Li
# --------------------------------------------------------

from skimage.draw import polygon
import numpy as np

def segToMask( S, h, w ):
    """
    Convert polygon segmentation to binary mask.
    :param   S (float array)   : polygon segmentation mask
    :param   h (int)           : target mask height
    :param   w (int)           : target mask width
    :return: M (bool 2D array) : binary mask
    """
    M = np.zeros((h,w), dtype=np.bool)
    for s in S:
        N = len(s)
        rr, cc = polygon(np.array(s[1:N:2]).clip(max=h-1), \
                      np.array(s[0:N:2]).clip(max=w-1)) # (y, x)
        M[rr, cc] = 1
    return M


def decodeMask(R):
    """
    Decode binary mask M encoded via run-length encoding.
    :param   R (object RLE)    : run-length encoding of binary mask
    :return: M (bool 2D array) : decoded binary mask
    """
    N = len(R['counts'])
    M = np.zeros( (R['size'][0]*R['size'][1], ))
    n = 0
    val = 1
    for pos in range(N):
        val = not val
        for c in range(R['counts'][pos]):
            R['counts'][pos]
            M[n] = val
            n += 1
    return M.reshape((R['size']), order='F')

def mask_coco2voc(coco_masks, im_height, im_width):
    voc_masks = np.zeros((len(coco_masks), im_height, im_width))
    for i, ann in enumerate(coco_masks):
        if type(ann) == list:
            # polygon
            m = segToMask(ann, im_height, im_width)
        else:
            # rle
            m = decodeMask(ann)
        voc_masks[i,:,:]=m;
    return voc_masks
