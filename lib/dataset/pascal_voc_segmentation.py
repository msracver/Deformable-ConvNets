"""
Pascal VOC Segmentation database
This class loads ground truth notations from standard Pascal VOC XML data formats
and transform them into IMDB format. Selective search is used for proposals, see segdb
function. Results are written as the Pascal VOC format. Evaluation is based on mAP
criterion.
"""

import cPickle
import os
import cv2
import numpy as np

from imdb import IMDB
from PIL import Image
from utils import image

class PascalVOC_Segmentation(IMDB):
    def __init__(self, image_set, root_path, devkit_path, result_path=None):
        """
        fill basic information to initialize imdb
        :param image_set: 2007_trainval, 2007_test, etc
        :param root_path: 'selective_search_data' and 'cache'
        :param devkit_path: data and results
        :return: imdb object
        """
        year, image_set = image_set.split('_', 1)
        super(PascalVOC_Segmentation, self).__init__('voc_' + year, image_set, root_path, devkit_path, result_path)  # set self.name

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

        self.config = {'comp_id': 'comp4',
                       'use_diff': False,
                       'min_size': 2}

    def load_image_set_index(self):
        """
        find out which indexes correspond to given image set (train or val)
        :return:
        """
        image_set_index_file = os.path.join(self.data_path, 'ImageSets', 'Segmentation', self.image_set + '.txt')
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

    def segmentation_class_path_from_index(self, index):
        """
        given image index, find out the full path of segmentation class
        :param index: index of a specific image
        :return: full path of segmentation class
        """
        seg_class_file = os.path.join(self.data_path, 'SegmentationClass', index + '.png')
        assert os.path.exists(seg_class_file), 'Path does not exist: {}'.format(seg_class_file)
        return seg_class_file

    def gt_segdb(self):
        """
        return ground truth image regions database
        :return: imdb[image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_segdb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                segdb = cPickle.load(fid)
            print '{} gt segdb loaded from {}'.format(self.name, cache_file)
            return segdb

        gt_segdb = [self.load_pascal_annotation(index) for index in self.image_set_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_segdb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt segdb to {}'.format(cache_file)

        return gt_segdb

    def load_pascal_annotation(self, index):
        """
        for a given index, load image and bounding boxes info from XML file
        :param index: index of a specific image
        :return: record['seg_cls_path', 'flipped']
        """
        import xml.etree.ElementTree as ET
        seg_rec = dict()
        seg_rec['image'] = self.image_path_from_index(index)
        size = cv2.imread(seg_rec['image']).shape
        seg_rec['height'] = size[0]
        seg_rec['width'] = size[1]

        seg_rec['seg_cls_path'] = self.segmentation_class_path_from_index(index)
        seg_rec['flipped'] = False

        return seg_rec

    def getpallete(self, num_cls):
        """
        this function is to get the colormap for visualizing the segmentation mask
        :param num_cls: the number of visulized class
        :return: the pallete
        """
        n = num_cls
        pallete = [0]*(n*3)
        for j in xrange(0,n):
                lab = j
                pallete[j*3+0] = 0
                pallete[j*3+1] = 0
                pallete[j*3+2] = 0
                i = 0
                while (lab > 0):
                        pallete[j*3+0] |= (((lab >> 0) & 1) << (7-i))
                        pallete[j*3+1] |= (((lab >> 1) & 1) << (7-i))
                        pallete[j*3+2] |= (((lab >> 2) & 1) << (7-i))
                        i = i + 1
                        lab >>= 3
        return pallete

    def evaluate_segmentations(self, pred_segmentations):
        """
        top level evaluations
        :param pred_segmentations: the pred segmentation result
        :return: the evaluation results
        """
        # make all these folders for results
        result_dir = os.path.join(self.result_path, 'results')
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        year_folder = os.path.join(self.result_path, 'results', 'VOC' + self.year)
        if not os.path.exists(year_folder):
            os.mkdir(year_folder)
        res_file_folder = os.path.join(self.result_path, 'results', 'VOC' + self.year, 'Segmentation')
        if not os.path.exists(res_file_folder):
            os.mkdir(res_file_folder)

        info = self.do_python_eval(pred_segmentations)
        self.write_segmentation_result(pred_segmentations, res_file_folder)
        return info

    def write_segmentation_result(self, pred_segmentations, res_file_folder):
        """
        Write pred segmentation to res_file_folder
        :param pred_segmentations: the pred segmentation results
        :param res_file_folder: the saving folder
        :return: [None]
        """
        pallete = self.getpallete(256)

        for i in range(len(pred_segmentations)):
            segmentation_result = np.uint8(np.squeeze(np.copy(pred_segmentations[i])))
            segmentation_result = Image.fromarray(segmentation_result)
            segmentation_result.putpalette(pallete)
            segmentation_result.save(os.path.join(res_file_folder, '%d_result.png'%(i)))

    def get_confusion_matrix(self, gt_label, pred_label, class_num):
        """
        Calcute the confusion matrix by given label and pred
        :param gt_label: the ground truth label
        :param pred_label: the pred label
        :param class_num: the nunber of class
        :return: the confusion matrix
        """
        index = (gt_label * class_num + pred_label).astype('int32')
        label_count = np.bincount(index)
        confusion_matrix = np.zeros((class_num, class_num))

        for i_label in range(class_num):
            for i_pred_label in range(class_num):
                cur_index = i_label * class_num + i_pred_label
                if cur_index < len(label_count):
                    confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

        return confusion_matrix

    def do_python_eval(self, pred_segmentations):
        """
        This function is a wrapper to calculte the metrics for given pred_segmentation results
        :param pred_segmentations: the pred segmentation result
        :return: the evaluation metrics
        """
        confusion_matrix = np.zeros((self.num_classes,self.num_classes))
        for i, index in enumerate(self.image_set_index):
            seg_gt_info = self.load_pascal_annotation(index)
            seg_gt = np.array(Image.open(seg_gt_info['seg_cls_path'])).astype('float32')
            seg_pred = np.squeeze(pred_segmentations[i])

            seg_gt = cv2.resize(seg_gt, (seg_pred.shape[1], seg_pred.shape[0]), interpolation=cv2.INTER_NEAREST)
            ignore_index = seg_gt != 255
            seg_gt = seg_gt[ignore_index]
            seg_pred = seg_pred[ignore_index]

            confusion_matrix += self.get_confusion_matrix(seg_gt, seg_pred, self.num_classes)

        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)

        mean_IU = (tp / np.maximum(1.0, pos + res - tp)).mean()

        return {'meanIU':mean_IU}
