#!/bin/bash

mkdir -p ./data
mkdir -p ./output
mkdir -p ./external/mxnet
mkdir -p ./model/pretrained_model

cd lib/bbox
python setup_linux.py build_ext --inplace
cd ../dataset/pycocotools
python setup_linux.py build_ext --inplace
cd ../../nms
python setup_linux.py build_ext --inplace
cd ../..
