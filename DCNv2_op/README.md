# DCNv2 operators

## Introduction

This folder provides the operators used in [DCNv2](https://arxiv.org/abs/1811.11168):

```
@article{DCNv2_2018,
  title={Deformable ConvNets v2: More Deformable, Better Results},
  author={Xizhou Zhu and Han Hu and Stephen Lin and Jifeng Dai},
  journal={arXiv:1811.11168},
  year={2018}
}
```

There are two operators in this folder. The first one is an updated deformable convolution operator. We have two major modifications:

* The new operator simutaneously processes multiple images in one computation loop, rather than processing one image in one loop as in the old operator.

    Both the old and new operators use the following computation pipeline (illustrated by a 3x3 deformable convolution with input data of NxCxHxW and output data of NxC'xHxW):

      for i in range(N/S):
          step 1 (slicing): slicing the input data at the batch dimension from i*S to (i+1)*S, input (NxCxHxW) -> sliced input (SxCxHxW)
          step 2 (deformable im2col): sliced input (SxCxHxW)+sliced offset (Sx18xHxW) -> column (Cx9xSxHxW)
          step 3 (MatMul&reshape): weight matrix (C'x 9C) * column (9CxSHW) -> temp sliced output (C'xSxHxW) -> sliced output (SxC'xHxW)
          step 4 (Merge): merge sliced output to form the whole output data (NxC'xHxW) 
      end

    In the old operator, S is fixed as 1. In the new operator, S can be set by a new *im2col_step* parameter its default value is min(N, 64). The new operator is significantly faster than the old one when the image batch size is large (e.g. 32 as a usual practice in ImageNet classification).

* The boundary processing scheme is modified.
    
    In the old operator, the pixel with any one coordinate (x or y) in (-inf, 0) is set as 0. The pixel with one coordinate in [H(W)-1,H(W)] is bilinear sampled assuming the pixel value on H(W) is the same as on H(W)-1. The pixel with any one coordinate in (H(W), inf) is set as 0.

    In the new operator, the input image is firstly padded with zeros and then bilinear sampling is performed on all range of locations.

    The new boundary scheme has little influence on tasks with large output feature map size (e.g. object detection), but can lead to better accuracy on tasks with small output feature map size (e.g. 7x7 as a usual practice in ImageNet classification).

The second operator is a new modulated deformable convolution operator introduced in the DCNv2 paper. Please see [example_symbol.py](https://github.com/msracver/Deformable-ConvNets/blob/master/DCNv2_op/example_symbol.py) for an example usage.
