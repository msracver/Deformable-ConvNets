# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuwen Xiong
# --------------------------------------------------------

import numpy as np
class Symbol:
    def __init__(self):
        self.arg_shape_dict = None
        self.out_shape_dict = None
        self.aux_shape_dict = None
        self.sym = None

    @property
    def symbol(self):
        return self.sym

    def get_symbol(self, cfg, is_train=True):
        """
        return a generated symbol, it also need to be assigned to self.sym
        """
        raise NotImplementedError()

    def init_weights(self, cfg, arg_params, aux_params):
        raise NotImplementedError()

    def get_msra_std(self, shape):
        fan_in = float(shape[1])
        if len(shape) > 2:
            fan_in *= np.prod(shape[2:])
        print(np.sqrt(2 / fan_in))
        return np.sqrt(2 / fan_in)

    def infer_shape(self, data_shape_dict):
        # infer shape
        arg_shape, out_shape, aux_shape = self.sym.infer_shape(**data_shape_dict)
        self.arg_shape_dict = dict(zip(self.sym.list_arguments(), arg_shape))
        self.out_shape_dict = dict(zip(self.sym.list_outputs(), out_shape))
        self.aux_shape_dict = dict(zip(self.sym.list_auxiliary_states(), aux_shape))

    def check_parameter_shapes(self, arg_params, aux_params, data_shape_dict, is_train=True):
        for k in self.sym.list_arguments():
            if k in data_shape_dict or (False if is_train else 'label' in k):
                continue
            assert k in arg_params, k + ' not initialized'
            assert arg_params[k].shape == self.arg_shape_dict[k], \
                'shape inconsistent for ' + k + ' inferred ' + str(self.arg_shape_dict[k]) + ' provided ' + str(
                    arg_params[k].shape)
        for k in self.sym.list_auxiliary_states():
            assert k in aux_params, k + ' not initialized'
            assert aux_params[k].shape == self.aux_shape_dict[k], \
                'shape inconsistent for ' + k + ' inferred ' + str(self.aux_shape_dict[k]) + ' provided ' + str(
                    aux_params[k].shape)
