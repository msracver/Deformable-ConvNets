# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li, Haocheng Zhang
# --------------------------------------------------------

import _init_paths

from xml.dom.minidom import parseString
import xml.etree.ElementTree as ET

import argparse
import os
import sys
import shutil
import logging
import pprint
import cv2
from config.config import config, update_config
from utils.image import resize, transform
import numpy as np
# get config
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
cur_path = os.path.abspath(os.path.dirname(__file__))
update_config(cur_path + '/../experiments/fpn/cfgs/resnet_v1_101_icdar_17_column_fpn_dcn_end2end_ohem.yaml')

sys.path.insert(0, os.path.join(cur_path, '../external/mxnet', config.MXNET_VERSION))
import mxnet as mx
from core.tester import im_detect, Predictor
from symbols import *
from utils.load_model import load_param
from utils.show_boxes import show_boxes
from utils.tictoc import tic, toc

from nms.nms import py_nms_wrapper, py_softnms_wrapper

IMAGE_EXTENSIONS = ['.jpg', '.png', '.bmp']
IoU_THRESHOLDS = ['0.5']

# set up class names
# CLASSES = ['table', 'figure', 'formula'] # TODOs
CLASSES = ['column']
CONCERNED_ERRORS = ['column']
WRITE_DETECTION_RESULTS = True

MODEL_EPOCH = 50
CONFIDENCE_THRESHOLD = 0.5
EXPERIMENT_NAME = "2cls-false_examples-" + str(MODEL_EPOCH) + "ep-" + str(CONFIDENCE_THRESHOLD) + "conf"
MODEL_PATH = './output/fpn/ICDAR_17_COLUMN/resnet_v1_101_icdar_17_column_fpn_dcn_end2end_ohem/train/fpn_icdar_17_column'


def bbox_intersection_over_union(boxA, boxB):
  # determine the (x, y)-coordinates of the intersection rectangle
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])
 
  # compute the area of intersection rectangle
  interArea = (xB - xA + 1) * (yB - yA + 1)
 
  # compute the area of both the prediction and ground-truth
  # rectangles
  boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
  boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
  # compute the intersection over union by taking the intersection
  # area and dividing it by the sum of prediction + ground-truth
  # areas - the interesection area
  iou = interArea / float(boxAArea + boxBArea - interArea)
 
  # return the intersection over union value
  return iou

def processCoordinates(coordinates):
    coords = coordinates.split(' ') # Separate based on space
    x1 = 10000
    x2 = 0
    y1= 10000
    y2 = 0
    for coord in coords:
        x, y = coord.split(',')
        x = float(x)
        y = float(y)
        x1 = min(x1, x)
        y1 = min(y1, y)
        x2 = max(x2, x)
        y2 = max(y2, y)

    # print (x1, y1, x2, y2)
    return x1, y1, x2, y2

def loadGTAnnotationsFromXML(xml_path):
    if not os.path.exists(xml_path):
        print ("Error: Unable to locate XML file %s" % (im_name))
        exit(-1)   
    tree = ET.parse(xml_path)
    objs = tree.findall('object')
    num_objs = len(objs)

    # Load object bounding boxes into a data frame.
    boundingBoxes = []
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text) - 1
        y1 = float(bbox.find('ymin').text) - 1
        x2 = float(bbox.find('xmax').text) - 1
        y2 = float(bbox.find('ymax').text) - 1
        #cls = self._class_to_ind[obj.find('name').text.lower().strip()]
        cls = obj.find('name').text.lower().strip()
        #objs =[obj for obj in objs if obj.find('name').text.lower().strip()]
        #cls = objs
                    
        if cls in CLASSES:
            boundingBoxes.append([x1, y1, x2, y2, 1.0, cls])
        #elif USE_REJECT_CLASS:
        #    boundingBoxes.append([x1, y1, x2, y2, 1.0, CLASSES[-1]]) # Add reject class
        #else:
        #    print ("Error: Class not found in list")
        #    exit (-1)

    return boundingBoxes

def convertToXML(im_name, bboxes):
    xml = '<document filename="' + im_name +'">'
    # for bbox in bboxes:
    for cls_idx, cls_name in enumerate(CLASSES):
        cls_dets = bboxes[cls_idx]
        for det in cls_dets:
            bbox = det[:4]
            # bbox[:4] = [int(coord) for coord in bbox[:4]]
            # Get document bounds
            topLeft = str(int(bbox[0])) + ',' + str(int(bbox[1]))
            topRight = str(int(bbox[2])) + ',' + str(int(bbox[1]))
            bottomLeft = str(int(bbox[0])) + ',' + str(int(bbox[3]))
            bottomRight = str(int(bbox[2])) + ',' + str(int(bbox[3]))
            
            xml += '<' + cls_name + 'Region prob="' + str(det[4]) + '">'
            xml += '<Coords points="' + topLeft + ' ' + topRight + ' ' + bottomLeft + ' ' + bottomRight + '"/>'
            xml += '</' + cls_name +'Region>'

    xml += '</document>'
    dom = parseString(xml)
    xml = dom.toprettyxml()
    xml = xml.split('\n')
    xml = xml[1:]
    xml = '\n'.join(xml)

    return xml    

def computeStatistics(detections, gt, statistics, iou_thresholds):
    classificationErrorMessage = ""
    for thresh in iou_thresholds:
        matchedGTBBox = [0] * len(gt)
        # Iterate over all the predicted bboxes
        for cls_idx, cls_name in enumerate(CLASSES):
            cls_dets = detections[cls_idx]
            for det in cls_dets:
                predictedBBox = det[:4]
                bboxMatchedIdx = -1
                for gtBBoxIdx, gtBBox in enumerate(gt):
                    # Compute IoU
                    iou = bbox_intersection_over_union(gtBBox, predictedBBox)
                    if ((iou > float(thresh)) and (gtBBox[5] == cls_name)):
                        if not matchedGTBBox[gtBBoxIdx]:
                            bboxMatchedIdx = gtBBoxIdx
                            break

                if (bboxMatchedIdx != -1):
                    statistics[cls_name][thresh]["truePositives"] += 1
                    matchedGTBBox[bboxMatchedIdx] = 1
                else:
                    statistics[cls_name][thresh]["falsePositives"] += 1
                    if cls_name in CONCERNED_ERRORS:
                        classificationErrorMessage += "[False Positive (Thresh: %s)] " % (thresh)

        # All the unmatched bboxes are false negatives
        for idx, gtBBox in enumerate(gt):
            if not matchedGTBBox[idx]:
                statistics[gtBBox[5]][thresh]["falseNegatives"] += 1
                if gtBBox[5] in CONCERNED_ERRORS:
                    classificationErrorMessage += "[False Negative (Thresh: %s)] " % (thresh)

    if len(classificationErrorMessage) == 0:
        classificationErrorMessage = None
    return statistics, classificationErrorMessage

def parse_args():
    parser = argparse.ArgumentParser(description='Show Deformable ConvNets demo')
    # general
    parser.add_argument('--rfcn_only', help='whether use R-FCN only (w/o Deformable ConvNets)', default=False, action='store_true')

    args = parser.parse_args()
    return args

args = parse_args()

def main():
    # get symbol
    pprint.pprint(config)
    config.symbol = 'resnet_v1_101_fpn_dcn_rcnn'
    sym_instance = eval(config.symbol + '.' + config.symbol)()
    sym = sym_instance.get_symbol(config, is_train=False)
    max_per_image = config.TEST.max_per_image

    # load demo data
    dataBaseDir = '/mnt/Imran/datasets/icdar_str_17/data/'
    outputBaseDir = '/mnt/Imran/StructureResults/ _ICDAR_17_COLUMN_Deformable-fpn-model-' + EXPERIMENT_NAME

    if os.path.exists(outputBaseDir):
        shutil.rmtree(outputBaseDir)
    os.mkdir(outputBaseDir)

    outputFile = open(os.path.join(outputBaseDir, 'output-' + EXPERIMENT_NAME + '.txt'), 'w')
    outputFile.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    errorStatsFile = open(os.path.join(outputBaseDir, 'inc-det-' + EXPERIMENT_NAME + '.txt'), 'w')

    incorrectDetectionResultsPath = os.path.join(outputBaseDir, 'IncorrectDetections-' + EXPERIMENT_NAME)
    if not os.path.exists(incorrectDetectionResultsPath):
        os.mkdir(incorrectDetectionResultsPath)

    detectionResultsPath = os.path.join(outputBaseDir, 'Detections-' + EXPERIMENT_NAME)
    if not os.path.exists(detectionResultsPath):
        os.mkdir(detectionResultsPath)

    statistics = {}
    for cls_ind, cls in enumerate(CLASSES):
        statistics[cls] = {}
        for thresh in IoU_THRESHOLDS:
            statistics[cls][thresh] = {}
            statistics[cls][thresh]["truePositives"] = 0
            statistics[cls][thresh]["falsePositives"] = 0
            statistics[cls][thresh]["falseNegatives"] = 0
            statistics[cls][thresh]["precision"] = 0
            statistics[cls][thresh]["recall"] = 0
            statistics[cls][thresh]["fMeasure"] = 0

    im_names_file = open(os.path.join(dataBaseDir, 'ImageSets/test.txt'), 'r')
    for im_name in im_names_file:
        data = []
        im_name = im_name.strip()
        # print ("Processing file: %s" % (im_name))

        found = False
        for ext in IMAGE_EXTENSIONS:
            im_name_with_ext = im_name + ext
            im_path = os.path.join(dataBaseDir, 'Images', im_name_with_ext)
            if os.path.exists(im_path):
                found = True
                break
        if not found:
            print ("Error: Unable to locate file %s" % (im_name))
            exit(-1)

        # Load GT annotations
        xml_path = os.path.join(dataBaseDir, 'Annotations', im_name + '.xml')
        gtBBoxes = loadGTAnnotationsFromXML(xml_path)

        im = cv2.imread(im_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        target_size = config.SCALES[0][0]
        max_size = config.SCALES[0][1]
        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
        data.append({'data': im_tensor, 'im_info': im_info})

        # get predictor
        data_names = ['data', 'im_info']
        label_names = []
        data = [[mx.nd.array(data[i][name]) for name in data_names] for i in xrange(len(data))]
        max_data_shape = [[('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]]
        provide_data = [[(k, v.shape) for k, v in zip(data_names, data[i])] for i in xrange(len(data))]
        provide_label = [None for i in xrange(len(data))]
        # arg_params, aux_params = load_param(cur_path + '/../model/' + ('rfcn_dcn_coco' if not args.rfcn_only else 'rfcn_coco'), 0, process=True)
        arg_params, aux_params = load_param(MODEL_PATH, MODEL_EPOCH, process=True)
        predictor = Predictor(sym, data_names, label_names,
                              context=[mx.gpu(0)], max_data_shapes=max_data_shape,
                              provide_data=provide_data, provide_label=provide_label,
                              arg_params=arg_params, aux_params=aux_params)

        # # warm up
        for j in xrange(2):
            data_batch = mx.io.DataBatch(data=[data[0]], label=[], pad=0, index=0,
                                         provide_data=[[(k, v.shape) for k, v in zip(data_names, data[0])]],
                                         provide_label=[None])
            scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]
            scores, boxes, data_dict = im_detect(predictor, data_batch, data_names, scales, config)

        # test
        image_names = [im_name] # Way around
        for idx, im_name in enumerate(image_names):
            data_batch = mx.io.DataBatch(data=[data[idx]], label=[], pad=0, index=idx,
                                         provide_data=[[(k, v.shape) for k, v in zip(data_names, data[idx])]],
                                         provide_label=[None])
            scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]

            tic()
            scores, boxes, data_dict = im_detect(predictor, data_batch, data_names, scales, config)
            boxes = boxes[0].astype('f')
            scores = scores[0].astype('f')
            dets_nms = []
            # TODO: Multi-scale testing
            for j in range(1, scores.shape[1]):
                cls_scores = scores[:, j, np.newaxis]
                cls_boxes = boxes[:, 4:8] if config.CLASS_AGNOSTIC else boxes[:, j * 4:(j + 1) * 4]
                cls_dets = np.hstack((cls_boxes, cls_scores))
                if config.TEST.USE_SOFTNMS:
                    soft_nms = py_softnms_wrapper(config.TEST.SOFTNMS_THRESH, max_dets=max_per_image)
                    cls_dets = soft_nms(cls_dets)
                else:
                    nms = py_nms_wrapper(config.TEST.NMS)
                    keep = nms(cls_dets)
                    cls_dets = cls_dets[keep, :]
                cls_dets = cls_dets[cls_dets[:, -1] > CONFIDENCE_THRESHOLD, :]
                dets_nms.append(cls_dets)

            # if max_per_image > 0:
            #     for idx_im in range(0, num_images):
            #         image_scores = np.hstack([all_boxes[j][idx_im][:, -1]
            #                                   for j in range(1, imdb.num_classes)])
            #         if len(image_scores) > max_per_image:
            #             image_thresh = np.sort(image_scores)[-max_per_image]
            #             for j in range(1, imdb.num_classes):
            #                 keep = np.where(all_boxes[j][idx_im][:, -1] >= image_thresh)[0]
            #                 all_boxes[j][idx_im] = all_boxes[j][idx_im][keep, :]

            print 'Processing image: {} {:.4f}s'.format(im_name, toc())

            # Add detections on the image
            im = cv2.imread(im_path) # Reload the image since the previous one was scaled
            for cls_idx, cls_name in enumerate(CONCERNED_ERRORS):
                cls_dets = dets_nms[cls_idx]
                for det in cls_dets:
                    predictedBBox = det[:4]
                    cv2.rectangle(im, (int(predictedBBox[0]), int(predictedBBox[1])), (int(predictedBBox[2]), int(predictedBBox[3])), (0, 0, 255), 2)
                    # w = predictedBBox[2] - predictedBBox[0]
                    # cv2.putText(im, cls_name, (int(predictedBBox[0] + (w / 2.0) - 100), int(predictedBBox[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            # Add gt annotations
            for bbox in gtBBoxes:
                if bbox[5] in CONCERNED_ERRORS:
                    cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)

            # Computate the statistics for the current image
            statistics, classificationErrorMessage = computeStatistics(dets_nms, gtBBoxes, statistics, IoU_THRESHOLDS)
            if classificationErrorMessage is not None:
                print ("Writing incorrect image: %s" % (im_name))
                errorStatsFile.write("%s: %s\n" % (im_name, classificationErrorMessage))
                cv2.imwrite(os.path.join(incorrectDetectionResultsPath, im_name + '.jpg'), im)

            # Write the output in ICDAR Format
            outputFile.write(convertToXML(im_name_with_ext, dets_nms))

            if WRITE_DETECTION_RESULTS:
                # visualize
                # im = cv2.imread(im_path)
                # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

                # # Get also the plot for saving on server
                # _, plt = show_boxes(im, dets_nms, CLASSES, 1, returnPlt=True)
                # plt.savefig(os.path.join(outputBaseDir, 'Detections', im_name[:im_name.rfind('.')] + ".png"))

                outputImagePath = os.path.join(detectionResultsPath, im_name + ".jpg")
                # print ("Writing image: %s" % (outputImagePath))
                cv2.imwrite(outputImagePath, im)

    outputFile.close()
    errorStatsFile.close()

    # Compute final precision and recall
    outputFile = open(os.path.join(outputBaseDir, 'output-stats-' + EXPERIMENT_NAME + '.txt'), 'w')
    for cls in statistics.keys():
        for thresh in statistics[cls].keys():
            if (statistics[cls][thresh]["truePositives"] == 0) and (statistics[cls][thresh]["falsePositives"] == 0):
                precision = 1.0
            else:
                precision = float(statistics[cls][thresh]["truePositives"]) / float(statistics[cls][thresh]["truePositives"] + statistics[cls][thresh]["falsePositives"])
            if (statistics[cls][thresh]["truePositives"] == 0) and (statistics[cls][thresh]["falseNegatives"] == 0):
                recall = 1.0
            else:
                recall = float(statistics[cls][thresh]["truePositives"]) / float(statistics[cls][thresh]["truePositives"] + statistics[cls][thresh]["falseNegatives"])
            if (precision == 0.0) and (recall == 0.0):
                fMeasure = 0.0
            else:
                fMeasure = 2 * ((precision * recall) / (precision + recall))

            statistics[cls][thresh]["precision"] = precision
            statistics[cls][thresh]["recall"] = recall
            statistics[cls][thresh]["fMeasure"] = fMeasure

            print ("--------------------------------")
            print ("Class: %s" % (cls))
            print ("IoU Threshold: %s" % (thresh))
            print ("True Positives: %d" % (statistics[cls][thresh]["truePositives"]))
            print ("False Positives: %d" % (statistics[cls][thresh]["falsePositives"]))
            print ("False Negatives: %d" % (statistics[cls][thresh]["falseNegatives"]))
            print ("Precision: %f" % (precision))
            print ("Recall: %f" % (recall))
            print ("F-Measure: %f" % (fMeasure))

            outputFile.write ("Class: %s" % (cls) + "\n")
            outputFile.write ("IoU Threshold: %s" % (thresh) + "\n")
            outputFile.write ("True Positives: %d" % (statistics[cls][thresh]["truePositives"]) + "\n")
            outputFile.write ("False Positives: %d" % (statistics[cls][thresh]["falsePositives"]) + "\n")
            outputFile.write ("False Negatives: %d" % (statistics[cls][thresh]["falseNegatives"]) + "\n")
            outputFile.write ("Precision: %f" % (precision) + "\n")
            outputFile.write ("Recall: %f" % (recall) + "\n")
            outputFile.write ("F-Measure: %f" % (fMeasure) + "\n")
            outputFile.write ("--------------------------------\n")

    outputFile.close()

if __name__ == '__main__':
    main()
