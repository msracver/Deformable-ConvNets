import numpy as np
import matplotlib.pyplot as plt
import random
import cv2

def show_masks(im, dets, msks, show = True, thresh = 1e-3, scale = 1.0):
    plt.cla()
    plt.imshow(im)
    for det, msk in zip(dets, msks):
        color = (random.random(), random.random(), random.random())  # generate a random color
        bbox = det[:4] * scale
        cod = np.zeros(4).astype(int)
        cod[0] = int(bbox[0])
        cod[1] = int(bbox[1])
        cod[2] = int(bbox[2])
        cod[3] = int(bbox[3])
        if im[cod[0]:cod[2], cod[1]:cod[3], 0].size > 0:
            msk = cv2.resize(msk, im[cod[1]:cod[3], cod[0]:cod[2], 0].T.shape)
            bimsk = msk > thresh
            bimsk = bimsk.astype(int)
            bimsk = np.repeat(bimsk[:, :, np.newaxis], 3, axis=2)
            mskd = im[cod[1]:cod[3], cod[0]:cod[2], :] * bimsk
            clmsk = np.ones(bimsk.shape) * bimsk
            clmsk[:, :, 0] = clmsk[:, :, 0] * color[0] * 256;
            clmsk[:, :, 1] = clmsk[:, :, 1] * color[1] * 256;
            clmsk[:, :, 2] = clmsk[:, :, 2] * color[2] * 256;
            im[cod[1]:cod[3], cod[0]:cod[2], :] = im[cod[1]:cod[3], cod[0]:cod[2], :] + 0.8 * clmsk - 0.8 * mskd
    plt.imshow(im)
    if(show):
        plt.show()
    return im

