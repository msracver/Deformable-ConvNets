# Singularity Recipe for MXNet

This is the recepe for MXNet and some packages needed to run R-FCN/Deformable R-FCN.

Since some of the supercomputing server only provides Singularity, so here we go.

# Software version of the container

- MXNet 1.1.0
- CUDA 9

# Usage

First of all (on your local machine), go the the root directory of the project, build the image with the following command:

    sudo singularity build --writable mxnet_rfcn.simg singularity/Singularity

Copy this image to the server where you want to run it. Also, you need to put the entire project {D-RFCN} onto the server as well (so that you can run your code)

On the server, change directory to the root of the project:

    module load cuda/9.0
    module load singularity
    singularity exec mxnet_rfcn.simg python experiments/fpn/fpn_test.py --cfg experiments/fpn/cfgs/resnet_v1_101_coco_trainval_fpn_dcn_end2end_ohem.yaml

If it all works fine, you can further count on sbatch to train/test the object detector.

REMEMBER to prepare the model and data and modify the path in `experiments/fpn/cfgs/resnet_v1_101_coco_trainval_fpn_dcn_end2end_ohem.yaml` so that `experiments/fpn/fpn_test.py` will work fine.