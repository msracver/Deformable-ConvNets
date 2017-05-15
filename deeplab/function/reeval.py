# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Zheng Zhang
# --------------------------------------------------------

import argparse
import cPickle
import os
import mxnet as mx

from config.config import config, generate_config
from dataset import *


def reeval(args):
    # load imdb
    imdb = eval(args.dataset)(args.image_set, args.root_path, args.dataset_path)

    # load detection results
    cache_file = os.path.join(imdb.cache_path, imdb.name, 'detections.pkl')
    with open(cache_file) as f:
        detections = cPickle.load(f)

    # eval
    imdb.evaluate_detections(detections)


def parse_args():
    parser = argparse.ArgumentParser(description='imdb test')
    # general
    parser.add_argument('--network', help='network name', default=default.network, type=str)
    parser.add_argument('--dataset', help='dataset name', default=default.dataset, type=str)
    args, rest = parser.parse_known_args()
    generate_config(args.network, args.dataset)
    parser.add_argument('--image_set', help='image_set name', default=default.image_set, type=str)
    parser.add_argument('--root_path', help='output data folder', default=default.root_path, type=str)
    parser.add_argument('--dataset_path', help='dataset path', default=default.dataset_path, type=str)
    # other
    parser.add_argument('--no_shuffle', help='disable random shuffle', action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print 'Called with argument:', args
    reeval(args)


if __name__ == '__main__':
    main()
