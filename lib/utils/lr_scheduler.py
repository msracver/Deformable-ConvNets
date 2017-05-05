# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yuwen Xiong
# --------------------------------------------------------


import logging
from mxnet.lr_scheduler import LRScheduler

class WarmupMultiFactorScheduler(LRScheduler):
    """Reduce learning rate in factor at steps specified in a list

    Assume the weight has been updated by n times, then the learning rate will
    be

    base_lr * factor^(sum((step/n)<=1)) # step is an array

    Parameters
    ----------
    step: list of int
        schedule learning rate after n updates
    factor: float
        the factor for reducing the learning rate
    """
    def __init__(self, step, factor=1, warmup=False, warmup_lr=0, warmup_step=0):
        super(WarmupMultiFactorScheduler, self).__init__()
        assert isinstance(step, list) and len(step) >= 1
        for i, _step in enumerate(step):
            if i != 0 and step[i] <= step[i-1]:
                raise ValueError("Schedule step must be an increasing integer list")
            if _step < 1:
                raise ValueError("Schedule step must be greater or equal than 1 round")
        if factor > 1.0:
            raise ValueError("Factor must be no more than 1 to make lr reduce")
        self.step = step
        self.cur_step_ind = 0
        self.factor = factor
        self.count = 0
        self.warmup = warmup
        self.warmup_lr = warmup_lr
        self.warmup_step = warmup_step

    def __call__(self, num_update):
        """
        Call to schedule current learning rate

        Parameters
        ----------
        num_update: int
            the maximal number of updates applied to a weight.
        """

        # NOTE: use while rather than if  (for continuing training via load_epoch)
        if self.warmup and num_update < self.warmup_step:
            return self.warmup_lr
        while self.cur_step_ind <= len(self.step)-1:
            if num_update > self.step[self.cur_step_ind]:
                self.count = self.step[self.cur_step_ind]
                self.cur_step_ind += 1
                self.base_lr *= self.factor
                logging.info("Update[%d]: Change learning rate to %0.5e",
                             num_update, self.base_lr)
            else:
                return self.base_lr
        return self.base_lr
