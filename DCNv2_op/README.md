# DCNv2 operators

## Introduction

This folder provides the operators used in [Deformable ConvNets v2](https://arxiv.org/abs/1811.11168):

```
@article{zhu2018deformable,
  title={Deformable ConvNets v2: More Deformable, Better Results},
  author={Zhu, Xizhou and Hu, Han and Lin, Stephen and Dai, Jifeng},
  journal={arXiv preprint arXiv:1811.11168},
  year={2018}
}
```

This folder provides the operators of modulated deformable convolution and RoIpooling. Especially, we updated the deformable conv layer which can reproduce the results in the Deformable ConvNets v2 paper. The major changes are as follows:

* To better handle occasions where sampling locations are outside of the image boundary.

    In the previous operator, if the sampling location is outside of the feature map boundary, its sampled value would be zero. Thus, the gradient with respect to learnable offset would be zero. We found such a scheme may deteriate the performance in ImageNet classification (perhaps because the feature maps are of low resolution). For object detection on COCO, both the previous and the updated operators deliver the same results.

    In the new operator, if the sampling location is within one pixel outside of the feature map boundary, bilinear sampling would also be applied. And gradient with respect to learnable offset can be non zero for such locations. This is implemented by padding zeros (by one row/column) outside of the boundaries of feature maps, and performing bilinear sampling on the padded feature maps.


* The efficiency of processing multiple images in a mini-batch is considerably improved.

    Both the previous and the updated operators follow the following computation pipeline (illustrated by a 3x3 deformable convolution with input data of NxCxHxW and output data of NxC'xHxW):

      for i in range(N/S):
          step 1 (slicing): slicing the input data at the batch dimension from i*S to (i+1)*S, input (NxCxHxW) -> sliced input (SxCxHxW)
          step 2 (deformable im2col): sliced input (SxCxHxW)+sliced offset (Sx18xHxW) -> column (Cx9xSxHxW)
          step 3 (MatMul&reshape): weight matrix (C'x 9C) * column (9CxSHW) -> temp sliced output (C'xSxHxW) -> sliced output (SxC'xHxW)
          step 4 (Merge): merge sliced output to form the whole output data (NxC'xHxW) 
      end

    In the previous operator, S is fixed as 1. In the updated operator, S can be set by the *im2col_step* parameter, whose default value is min(N, 64). The updated operator is significantly faster than the existing one when the image batch size is large.


Check [example_symbol.py](https://github.com/msracver/Deformable-ConvNets/blob/master/DCNv2_op/example_symbol.py) for an example of utilizing the operators.
