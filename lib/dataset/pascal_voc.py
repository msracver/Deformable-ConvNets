# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Haozhi Qi, from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# --------------------------------------------------------

"""
Pascal VOC database
This class loads ground truth notations from standard Pascal VOC XML data formats
and transform them into IMDB format. Selective search is used for proposals, see roidb
function. Results are written as the Pascal VOC format. Evaluation is based on mAP
criterion.
"""

import cPickle
import cv2
import os
import numpy as np
import PIL

from imdb import IMDB
from pascal_voc_eval import voc_eval, voc_eval_sds
from ds_utils import unique_boxes, filter_small_boxes


class PascalVOC(IMDB):
    def __init__(self, image_set, root_path, devkit_path, result_path=None, mask_size=-1, binary_thresh=None):
        """
        fill basic information to initialize imdb
        :param image_set: 2007_trainval, 2007_test, etc
        :param root_path: 'selective_search_data' and 'cache'
        :param devkit_path: data and results
        :return: imdb object
        """
        year, image_set = image_set.split('_')
        super(PascalVOC, self).__init__('voc_' + year, image_set, root_path, devkit_path, result_path)  # set self.name

        self.year = year
        self.root_path = root_path
        self.devkit_path = devkit_path
        self.data_path = os.path.join(devkit_path, 'VOC' + year)

        self.classes = ['__background__',  # always index 0
                        'aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse',
                        'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor']
        self.num_classes = len(self.classes)
        self.image_set_index = self.load_image_set_index()
        self.num_images = len(self.image_set_index)
        print 'num_images', self.num_images
        self.mask_size = mask_size
        self.binary_thresh = binary_thresh

        self.config = {'comp_id': 'comp4',
                       'use_diff': False,
                       'min_size': 2}

    def load_image_set_index(self):
        """
        find out which indexes correspond to given image set (train or val)
        :return:
        """
        image_set_index_file = os.path.join(self.data_path, 'ImageSets', 'Main', self.image_set + '.txt')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file) as f:
            image_set_index = [x.strip() for x in f.readlines()]
        return image_set_index

    def image_path_from_index(self, index):
        """
        given image index, find out full path
        :param index: index of a specific image
        :return: full path of this image
        """
        image_file = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file


    def gt_roidb(self):
        """
        return ground truth image regions database
        :return: imdb[image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self.load_pascal_annotation(index) for index in self.image_set_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def load_pascal_annotation(self, index):
        """
        for a given index, load image and bounding boxes info from XML file
        :param index: index of a specific image
        :return: record['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        import xml.etree.ElementTree as ET
        roi_rec = dict()
        roi_rec['image'] = self.image_path_from_index(index)

        filename = os.path.join(self.data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        size = tree.find('size')
        roi_rec['height'] = float(size.find('height').text)
        roi_rec['width'] = float(size.find('width').text)
        #im_size = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION).shape
        #assert im_size[0] == roi_rec['height'] and im_size[1] == roi_rec['width']

        objs = tree.findall('object')
        if not self.config['use_diff']:
            non_diff_objs = [obj for obj in objs if int(obj.find('difficult').text) == 0]
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        class_to_index = dict(zip(self.classes, range(self.num_classes)))
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = class_to_index[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        roi_rec.update({'boxes': boxes,
                        'gt_classes': gt_classes,
                        'gt_overlaps': overlaps,
                        'max_classes': overlaps.argmax(axis=1),
                        'max_overlaps': overlaps.max(axis=1),
                        'flipped': False})
        return roi_rec

    def load_selective_search_roidb(self, gt_roidb):
        """
        turn selective search proposals into selective search roidb
        :param gt_roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        :return: roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        import scipy.io
        matfile = os.path.join(self.root_path, 'selective_search_data', self.name + '.mat')
        assert os.path.exists(matfile), 'selective search data does not exist: {}'.format(matfile)
        raw_data = scipy.io.loadmat(matfile)['boxes'].ravel()  # original was dict ['images', 'boxes']

        box_list = []
        for i in range(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1  # pascal voc dataset starts from 1.
            keep = unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def selective_search_roidb(self, gt_roidb, append_gt=False):
        """
        get selective search roidb and ground truth roidb
        :param gt_roidb: ground truth roidb
        :param append_gt: append ground truth
        :return: roidb of selective search
        """
        cache_file = os.path.join(self.cache_path, self.name + '_ss_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if append_gt:
            print 'appending ground truth annotations'
            ss_roidb = self.load_selective_search_roidb(gt_roidb)
            roidb = IMDB.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self.load_selective_search_roidb(gt_roidb)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def evaluate_detections(self, detections):
        """
        top level evaluations
        :param detections: result matrix, [bbox, confidence]
        :return: None
        """
        # make all these folders for results
        result_dir = os.path.join(self.result_path, 'results')
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        year_folder = os.path.join(self.result_path, 'results', 'VOC' + self.year)
        if not os.path.exists(year_folder):
            os.mkdir(year_folder)
        res_file_folder = os.path.join(self.result_path, 'results', 'VOC' + self.year, 'Main')
        if not os.path.exists(res_file_folder):
            os.mkdir(res_file_folder)

        self.write_pascal_results(detections)
        info = self.do_python_eval()
        return info

    def evaluate_sds(self, all_boxes, all_masks):
        self._write_voc_seg_results_file(all_boxes, all_masks)
        info = self._py_evaluate_segmentation()
        return info

    def _write_voc_seg_results_file(self, all_boxes, all_masks):
        """
        Write results as a pkl file, note this is different from
        detection task since it's difficult to write masks to txt
        """
        # make all these folders for results
        result_dir = os.path.join(self.result_path, 'results')
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        # Always reformat result in case of sometimes masks are not
        # binary or is in shape (n, sz*sz) instead of (n, sz, sz)
        all_boxes, all_masks = self._reformat_result(all_boxes, all_masks)
        for cls_inds, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = os.path.join(result_dir, cls + '_det.pkl')
            print filename
            with open(filename, 'wb') as f:
                cPickle.dump(all_boxes[cls_inds], f, cPickle.HIGHEST_PROTOCOL)
            filename = os.path.join(result_dir, cls + '_seg.pkl')
            with open(filename, 'wb') as f:
                cPickle.dump(all_masks[cls_inds], f, cPickle.HIGHEST_PROTOCOL)

    def _reformat_result(self, boxes, masks):
        num_images = self.num_images
        num_class = len(self.classes)
        reformat_masks = [[[] for _ in xrange(num_images)]
                          for _ in xrange(num_class)]
        for cls_inds in xrange(1, num_class):
            for img_inds in xrange(num_images):
                if len(masks[cls_inds][img_inds]) == 0:
                    continue
                num_inst = masks[cls_inds][img_inds].shape[0]
                #print num_inst
                #print masks[cls_inds][img_inds].shape
                reformat_masks[cls_inds][img_inds] = masks[cls_inds][img_inds]\
                    .reshape(num_inst, self.mask_size, self.mask_size)
                # reformat_masks[cls_inds][img_inds] = reformat_masks[cls_inds][img_inds] >= 0.4
        all_masks = reformat_masks
        return boxes, all_masks

    def _py_evaluate_segmentation(self):
        info_str = ''
        gt_dir = self.data_path
        imageset_file = os.path.join(self.data_path, 'ImageSets', 'Main', self.image_set + '.txt')
        cache_dir = os.path.join(self.devkit_path, 'annotations_cache')
        output_dir = os.path.join(self.result_path, 'results')
        aps = []
        # define this as true according to SDS's evaluation protocol
        use_07_metric = True
        print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
        info_str += 'VOC07 metric? ' + ('Y' if use_07_metric else 'No')
        info_str += '\n'
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        print '~~~~~~ Evaluation use min overlap = 0.5 ~~~~~~'
        info_str += '~~~~~~ Evaluation use min overlap = 0.5 ~~~~~~'
        info_str += '\n'
        for i, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            det_filename = os.path.join(output_dir, cls + '_det.pkl')
            seg_filename = os.path.join(output_dir, cls + '_seg.pkl')
            ap = voc_eval_sds(det_filename, seg_filename, gt_dir,
                              imageset_file, cls, cache_dir, self.classes, self.mask_size, self.binary_thresh, ov_thresh=0.5)
            aps += [ap]
            print('AP for {} = {:.2f}'.format(cls, ap*100))
            info_str += 'AP for {} = {:.2f}\n'.format(cls, ap*100)
        print('Mean AP@0.5 = {:.2f}'.format(np.mean(aps)*100))
        info_str += 'Mean AP@0.5 = {:.2f}\n'.format(np.mean(aps)*100)
        print '~~~~~~ Evaluation use min overlap = 0.7 ~~~~~~'
        info_str += '~~~~~~ Evaluation use min overlap = 0.7 ~~~~~~\n'
        aps = []
        for i, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            det_filename = os.path.join(output_dir, cls + '_det.pkl')
            seg_filename = os.path.join(output_dir, cls + '_seg.pkl')
            ap = voc_eval_sds(det_filename, seg_filename, gt_dir,
                              imageset_file, cls, cache_dir, self.classes, self.mask_size, self.binary_thresh, ov_thresh=0.7)
            aps += [ap]
            print('AP for {} = {:.2f}'.format(cls, ap*100))
            info_str += 'AP for {} = {:.2f}\n'.format(cls, ap*100)
        print('Mean AP@0.7 = {:.2f}'.format(np.mean(aps)*100))
        info_str += 'Mean AP@0.7 = {:.2f}\n'.format(np.mean(aps)*100)

        return info_str

    def get_result_file_template(self):
        """
        this is a template
        VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        :return: a string template
        """
        res_file_folder = os.path.join(self.result_path, 'results', 'VOC' + self.year, 'Main')
        comp_id = self.config['comp_id']
        filename = comp_id + '_det_' + self.image_set + '_{:s}.txt'
        path = os.path.join(res_file_folder, filename)
        return path

    def write_pascal_results(self, all_boxes):
        """
        write results files in pascal devkit path
        :param all_boxes: boxes to be processed [bbox, confidence]
        :return: None
        """
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = self.get_result_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_set_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if len(dets) == 0:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1, dets[k, 2] + 1, dets[k, 3] + 1))

    def do_python_eval(self):
        """
        python evaluation wrapper
        :return: info_str
        """
        info_str = ''
        annopath = os.path.join(self.data_path, 'Annotations', '{0!s}.xml')
        imageset_file = os.path.join(self.data_path, 'ImageSets', 'Main', self.image_set + '.txt')
        annocache = os.path.join(self.cache_path, self.name + '_annotations.pkl')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if self.year == 'SDS' or int(self.year) < 2010 else False
        print 'VOC07 metric? ' + ('Y' if use_07_metric else 'No')
        info_str += 'VOC07 metric? ' + ('Y' if use_07_metric else 'No')
        info_str += '\n'
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            filename = self.get_result_file_template().format(cls)
            rec, prec, ap = voc_eval(filename, annopath, imageset_file, cls, annocache,
                                     ovthresh=0.5, use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            info_str += 'AP for {} = {:.4f}\n'.format(cls, ap)
        print('Mean AP@0.5 = {:.4f}'.format(np.mean(aps)))
        info_str += 'Mean AP@0.5 = {:.4f}\n\n'.format(np.mean(aps))
        # @0.7
        aps = []
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            filename = self.get_result_file_template().format(cls)
            rec, prec, ap = voc_eval(filename, annopath, imageset_file, cls, annocache,
                                     ovthresh=0.7, use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            info_str += 'AP for {} = {:.4f}\n'.format(cls, ap)
        print('Mean AP@0.7 = {:.4f}'.format(np.mean(aps)))
        info_str += 'Mean AP@0.7 = {:.4f}'.format(np.mean(aps))
        return info_str
