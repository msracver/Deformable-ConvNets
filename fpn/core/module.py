# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Zheng Zhang
# --------------------------------------------------------
# Based on:
# MX-RCNN
# Copyright (c) 2016 by Contributors
# Licence under The Apache 2.0 License
# https://github.com/ijkguo/mx-rcnn/
# --------------------------------------------------------

"""A `MutableModule` implement the `BaseModule` API, and allows input shape
varying with training iterations. If shapes vary, executors will rebind,
using shared arrays from the initial module binded with maximum shape.
"""

import time
import logging
import warnings

from mxnet import context as ctx
from mxnet.initializer import Uniform, InitDesc
from mxnet.module.base_module import BaseModule, _check_input_names, _parse_data_desc, _as_list
from mxnet.model import _create_kvstore, _initialize_kvstore, _update_params, _update_params_on_kvstore, load_checkpoint, BatchEndParam
from mxnet import metric
# from mxnet.module.executor_group import DataParallelExecutorGroup

from .DataParallelExecutorGroup import DataParallelExecutorGroup
from mxnet import ndarray as nd
from mxnet import optimizer as opt


class Module(BaseModule):
    """Module is a basic module that wrap a `Symbol`. It is functionally the same
    as the `FeedForward` model, except under the module API.

    Parameters
    ----------
    symbol : Symbol
    data_names : list of str
        Default is `('data')` for a typical model used in image classification.
    label_names : list of str
        Default is `('softmax_label')` for a typical model used in image
        classification.
    logger : Logger
        Default is `logging`.
    context : Context or list of Context
        Default is `cpu()`.
    work_load_list : list of number
        Default `None`, indicating uniform workload.
    fixed_param_names: list of str
        Default `None`, indicating no network parameters are fixed.
    state_names : list of str
        states are similar to data and label, but not provided by data iterator.
        Instead they are initialized to 0 and can be set by set_states()
    """
    def __init__(self, symbol, data_names=('data',), label_names=('softmax_label',),
                 logger=logging, context=ctx.cpu(), work_load_list=None,
                 fixed_param_names=None, state_names=None):
        super(Module, self).__init__(logger=logger)

        if isinstance(context, ctx.Context):
            context = [context]
        self._context = context
        if work_load_list is None:
            work_load_list = [1] * len(self._context)
        assert len(work_load_list) == len(self._context)
        self._work_load_list = work_load_list

        self._symbol = symbol

        data_names = list(data_names) if data_names is not None else []
        label_names = list(label_names) if label_names is not None else []
        state_names = list(state_names) if state_names is not None else []
        fixed_param_names = list(fixed_param_names) if fixed_param_names is not None else []

        _check_input_names(symbol, data_names, "data", True)
        _check_input_names(symbol, label_names, "label", False)
        _check_input_names(symbol, state_names, "state", True)
        _check_input_names(symbol, fixed_param_names, "fixed_param", True)

        arg_names = symbol.list_arguments()
        input_names = data_names + label_names + state_names
        self._param_names = [x for x in arg_names if x not in input_names]
        self._fixed_param_names = fixed_param_names
        self._aux_names = symbol.list_auxiliary_states()
        self._data_names = data_names
        self._label_names = label_names
        self._state_names = state_names
        self._output_names = symbol.list_outputs()

        self._arg_params = None
        self._aux_params = None
        self._params_dirty = False

        self._optimizer = None
        self._kvstore = None
        self._update_on_kvstore = None
        self._updater = None
        self._preload_opt_states = None
        self._grad_req = None

        self._exec_group = None
        self._data_shapes = None
        self._label_shapes = None

    @staticmethod
    def load(prefix, epoch, load_optimizer_states=False, **kwargs):
        """Create a model from previously saved checkpoint.

        Parameters
        ----------
        prefix : str
            path prefix of saved model files. You should have
            "prefix-symbol.json", "prefix-xxxx.params", and
            optionally "prefix-xxxx.states", where xxxx is the
            epoch number.
        epoch : int
            epoch to load.
        load_optimizer_states : bool
            whether to load optimizer states. Checkpoint needs
            to have been made with save_optimizer_states=True.
        data_names : list of str
            Default is `('data')` for a typical model used in image classification.
        label_names : list of str
            Default is `('softmax_label')` for a typical model used in image
            classification.
        logger : Logger
            Default is `logging`.
        context : Context or list of Context
            Default is `cpu()`.
        work_load_list : list of number
            Default `None`, indicating uniform workload.
        fixed_param_names: list of str
            Default `None`, indicating no network parameters are fixed.
        """
        sym, args, auxs = load_checkpoint(prefix, epoch)
        mod = Module(symbol=sym, **kwargs)
        mod._arg_params = args
        mod._aux_params = auxs
        mod.params_initialized = True
        if load_optimizer_states:
            mod._preload_opt_states = '%s-%04d.states'%(prefix, epoch)
        return mod

    def save_checkpoint(self, prefix, epoch, save_optimizer_states=False):
        """Save current progress to checkpoint.
        Use mx.callback.module_checkpoint as epoch_end_callback to save during training.

        Parameters
        ----------
        prefix : str
            The file prefix to checkpoint to
        epoch : int
            The current epoch number
        save_optimizer_states : bool
            Whether to save optimizer states for continue training
        """
        self._symbol.save('%s-symbol.json'%prefix)
        param_name = '%s-%04d.params' % (prefix, epoch)
        self.save_params(param_name)
        logging.info('Saved checkpoint to \"%s\"', param_name)
        if save_optimizer_states:
            state_name = '%s-%04d.states' % (prefix, epoch)
            self.save_optimizer_states(state_name)
            logging.info('Saved optimizer state to \"%s\"', state_name)

    def _reset_bind(self):
        """Internal function to reset binded state."""
        self.binded = False
        self._exec_group = None
        self._data_shapes = None
        self._label_shapes = None

    @property
    def data_names(self):
        """A list of names for data required by this module."""
        return self._data_names

    @property
    def label_names(self):
        """A list of names for labels required by this module."""
        return self._label_names

    @property
    def output_names(self):
        """A list of names for the outputs of this module."""
        return self._output_names

    @property
    def data_shapes(self):
        """Get data shapes.
        Returns
        -------
        A list of `(name, shape)` pairs.
        """
        assert self.binded
        return self._data_shapes

    @property
    def label_shapes(self):
        """Get label shapes.
        Returns
        -------
        A list of `(name, shape)` pairs. The return value could be `None` if
        the module does not need labels, or if the module is not binded for
        training (in this case, label information is not available).
        """
        assert self.binded
        return self._label_shapes

    @property
    def output_shapes(self):
        """Get output shapes.
        Returns
        -------
        A list of `(name, shape)` pairs.
        """
        assert self.binded
        return self._exec_group.get_output_shapes()

    def get_params(self):
        """Get current parameters.
        Returns
        -------
        `(arg_params, aux_params)`, each a dictionary of name to parameters (in
        `NDArray`) mapping.
        """
        assert self.binded and self.params_initialized

        if self._params_dirty:
            self._sync_params_from_devices()
        return (self._arg_params, self._aux_params)

    def init_params(self, initializer=Uniform(0.01), arg_params=None, aux_params=None,
                    allow_missing=False, force_init=False, allow_extra=False):
        """Initialize the parameters and auxiliary states.

        Parameters
        ----------
        initializer : Initializer
            Called to initialize parameters if needed.
        arg_params : dict
            If not None, should be a dictionary of existing arg_params. Initialization
            will be copied from that.
        aux_params : dict
            If not None, should be a dictionary of existing aux_params. Initialization
            will be copied from that.
        allow_missing : bool
            If true, params could contain missing values, and the initializer will be
            called to fill those missing params.
        force_init : bool
            If true, will force re-initialize even if already initialized.
        """
        if self.params_initialized and not force_init:
            warnings.warn("Parameters already initialized and force_init=False. "
                          "init_params call ignored.", stacklevel=2)
            return
        assert self.binded, 'call bind before initializing the parameters'

        def _impl(name, arr, cache):
            """Internal helper for parameter initialization"""
            if cache is not None:
                if name in cache:
                    cache_arr = cache[name]

                    # just in case the cached array is just the target itself
                    if cache_arr is not arr:
                        cache_arr.copyto(arr)
                else:
                    if not allow_missing:
                        raise RuntimeError("%s is not presented" % name)
                    if initializer != None:
                        initializer(name, arr)
            else:
                initializer(name, arr)

        attrs = self._symbol.attr_dict()
        for name, arr in self._arg_params.items():
            desc = InitDesc(name, attrs.get(name, None))
            _impl(desc, arr, arg_params)

        for name, arr in self._aux_params.items():
            desc = InitDesc(name, attrs.get(name, None))
            _impl(desc, arr, aux_params)

        self.params_initialized = True
        self._params_dirty = False

        # copy the initialized parameters to devices
        self._exec_group.set_params(self._arg_params, self._aux_params)

    def set_params(self, arg_params, aux_params, allow_missing=False, force_init=True):
        """Assign parameter and aux state values.

        Parameters
        ----------
        arg_params : dict
            Dictionary of name to value (`NDArray`) mapping.
        aux_params : dict
            Dictionary of name to value (`NDArray`) mapping.
        allow_missing : bool
            If true, params could contain missing values, and the initializer will be
            called to fill those missing params.
        force_init : bool
            If true, will force re-initialize even if already initialized.

        Examples
        --------
        An example of setting module parameters::
            >>> sym, arg_params, aux_params = \
            >>>     mx.model.load_checkpoint(model_prefix, n_epoch_load)
            >>> mod.set_params(arg_params=arg_params, aux_params=aux_params)
        """
        if not allow_missing:
            self.init_params(initializer=None, arg_params=arg_params, aux_params=aux_params,
                             allow_missing=allow_missing, force_init=force_init)
            return

        if self.params_initialized and not force_init:
            warnings.warn("Parameters already initialized and force_init=False. "
                          "set_params call ignored.", stacklevel=2)
            return

        self._exec_group.set_params(arg_params, aux_params)

        # because we didn't update self._arg_params, they are dirty now.
        self._params_dirty = True
        self.params_initialized = True

    def bind(self, data_shapes, label_shapes=None, for_training=True,
             inputs_need_grad=False, force_rebind=False, shared_module=None,
             grad_req='write'):
        """Bind the symbols to construct executors. This is necessary before one
        can perform computation with the module.

        Parameters
        ----------
        data_shapes : list of (str, tuple)
            Typically is `data_iter.provide_data`.
        label_shapes : list of (str, tuple)
            Typically is `data_iter.provide_label`.
        for_training : bool
            Default is `True`. Whether the executors should be bind for training.
        inputs_need_grad : bool
            Default is `False`. Whether the gradients to the input data need to be computed.
            Typically this is not needed. But this might be needed when implementing composition
            of modules.
        force_rebind : bool
            Default is `False`. This function does nothing if the executors are already
            binded. But with this `True`, the executors will be forced to rebind.
        shared_module : Module
            Default is `None`. This is used in bucketing. When not `None`, the shared module
            essentially corresponds to a different bucket -- a module with different symbol
            but with the same sets of parameters (e.g. unrolled RNNs with different lengths).
        """
        # force rebinding is typically used when one want to switch from
        # training to prediction phase.
        if force_rebind:
            self._reset_bind()

        if self.binded:
            self.logger.warning('Already binded, ignoring bind()')
            return

        self.for_training = for_training
        self.inputs_need_grad = inputs_need_grad
        self.binded = True
        self._grad_req = grad_req

        if not for_training:
            assert not inputs_need_grad
        else:
            pass
            # this is not True, as some module might not contains a loss function
            # that consumes the labels
            # assert label_shapes is not None

        # self._data_shapes, self._label_shapes = _parse_data_desc(
        #     self.data_names, self.label_names, data_shapes, label_shapes)
        self._data_shapes, self._label_shapes = zip(*[_parse_data_desc(self.data_names, self.label_names, data_shape, label_shape)
                                                      for data_shape, label_shape in zip(data_shapes, label_shapes)])
        if self._label_shapes.count(None) == len(self._label_shapes):
            self._label_shapes = None

        if shared_module is not None:
            assert isinstance(shared_module, Module) and \
                    shared_module.binded and shared_module.params_initialized
            shared_group = shared_module._exec_group
        else:
            shared_group = None
        self._exec_group = DataParallelExecutorGroup(self._symbol, self._context,
                                                     self._work_load_list, self._data_shapes,
                                                     self._label_shapes, self._param_names,
                                                     for_training, inputs_need_grad,
                                                     shared_group, logger=self.logger,
                                                     fixed_param_names=self._fixed_param_names,
                                                     grad_req=grad_req,
                                                     state_names=self._state_names)
        # self._total_exec_bytes = self._exec_group._total_exec_bytes
        if shared_module is not None:
            self.params_initialized = True
            self._arg_params = shared_module._arg_params
            self._aux_params = shared_module._aux_params
        elif self.params_initialized:
            # if the parameters are already initialized, we are re-binding
            # so automatically copy the already initialized params
            self._exec_group.set_params(self._arg_params, self._aux_params)
        else:
            assert self._arg_params is None and self._aux_params is None
            param_arrays = [
                nd.zeros(x[0].shape, dtype=x[0].dtype)
                for x in self._exec_group.param_arrays
            ]
            self._arg_params = {name:arr for name, arr in zip(self._param_names, param_arrays)}

            aux_arrays = [
                nd.zeros(x[0].shape, dtype=x[0].dtype)
                for x in self._exec_group.aux_arrays
            ]
            self._aux_params = {name:arr for name, arr in zip(self._aux_names, aux_arrays)}

        if shared_module is not None and shared_module.optimizer_initialized:
            self.borrow_optimizer(shared_module)


    def reshape(self, data_shapes, label_shapes=None):
        """Reshape the module for new input shapes.

        Parameters
        ----------
        data_shapes : list of (str, tuple)
            Typically is `data_iter.provide_data`.
        label_shapes : list of (str, tuple)
            Typically is `data_iter.provide_label`.
        """
        assert self.binded
        # self._data_shapes, self._label_shapes = _parse_data_desc(
        #     self.data_names, self.label_names, data_shapes, label_shapes)
        self._data_shapes, self._label_shapes = zip(*[_parse_data_desc(self.data_names, self.label_names, data_shape, label_shape)
                                                      for data_shape, label_shape in zip(data_shapes, label_shapes)])

        self._exec_group.reshape(self._data_shapes, self._label_shapes)


    def init_optimizer(self, kvstore='local', optimizer='sgd',
                       optimizer_params=(('learning_rate', 0.01),), force_init=False):
        """Install and initialize optimizers.

        Parameters
        ----------
        kvstore : str or KVStore
            Default `'local'`.
        optimizer : str or Optimizer
            Default `'sgd'`
        optimizer_params : dict
            Default `(('learning_rate', 0.01),)`. The default value is not a dictionary,
            just to avoid pylint warning of dangerous default values.
        force_init : bool
            Default `False`, indicating whether we should force re-initializing the
            optimizer in the case an optimizer is already installed.
        """
        assert self.binded and self.params_initialized

        if self.optimizer_initialized and not force_init:
            self.logger.warning('optimizer already initialized, ignoring...')
            return

        (kvstore, update_on_kvstore) = \
                _create_kvstore(kvstore, len(self._context), self._arg_params)

        batch_size = self._exec_group.batch_size
        if kvstore and 'dist' in kvstore.type and '_sync' in kvstore.type:
            batch_size *= kvstore.num_workers
        rescale_grad = 1.0/batch_size

        if isinstance(optimizer, str):
            idx2name = {}
            if update_on_kvstore:
                idx2name.update(enumerate(self._exec_group.param_names))
            else:
                for k in range(len(self._context)):
                    idx2name.update({i*len(self._context)+k: n
                                     for i, n in enumerate(self._exec_group.param_names)})
            optimizer_params = dict(optimizer_params)
            if 'rescale_grad' not in optimizer_params:
                optimizer_params['rescale_grad'] = rescale_grad
            optimizer = opt.create(optimizer,
                                   sym=self.symbol, param_idx2name=idx2name,
                                   **optimizer_params)
        else:
            assert isinstance(optimizer, opt.Optimizer)
            if optimizer.rescale_grad != rescale_grad:
                #pylint: disable=no-member
                warnings.warn(
                    "Optimizer created manually outside Module but rescale_grad " +
                    "is not normalized to 1.0/batch_size/num_workers (%s vs. %s). "%(
                        optimizer.rescale_grad, rescale_grad) +
                    "Is this intended?", stacklevel=2)

        self._optimizer = optimizer
        self._kvstore = kvstore
        self._update_on_kvstore = update_on_kvstore
        self._updater = None

        if kvstore:
            # copy initialized local parameters to kvstore
            _initialize_kvstore(kvstore=kvstore,
                                param_arrays=self._exec_group.param_arrays,
                                arg_params=self._arg_params,
                                param_names=self._param_names,
                                update_on_kvstore=update_on_kvstore)
        if update_on_kvstore:
            kvstore.set_optimizer(self._optimizer)
        else:
            self._updater = opt.get_updater(optimizer)

        self.optimizer_initialized = True

        if self._preload_opt_states is not None:
            self.load_optimizer_states(self._preload_opt_states)
            self._preload_opt_states = None

    def borrow_optimizer(self, shared_module):
        """Borrow optimizer from a shared module. Used in bucketing, where exactly the same
        optimizer (esp. kvstore) is used.

        Parameters
        ----------
        shared_module : Module
        """
        assert shared_module.optimizer_initialized
        self._optimizer = shared_module._optimizer
        self._kvstore = shared_module._kvstore
        self._update_on_kvstore = shared_module._update_on_kvstore
        self._updater = shared_module._updater
        self.optimizer_initialized = True

    def forward(self, data_batch, is_train=None):
        """Forward computation.

        Parameters
        ----------
        data_batch : DataBatch
            Could be anything with similar API implemented.
        is_train : bool
            Default is `None`, which means `is_train` takes the value of `self.for_training`.
        """
        assert self.binded and self.params_initialized
        self._exec_group.forward(data_batch, is_train)

    def backward(self, out_grads=None):
        """Backward computation.

        Parameters
        ----------
        out_grads : NDArray or list of NDArray, optional
            Gradient on the outputs to be propagated back.
            This parameter is only needed when bind is called
            on outputs that are not a loss function.
        """
        assert self.binded and self.params_initialized
        self._exec_group.backward(out_grads=out_grads)

    def update(self):
        """Update parameters according to the installed optimizer and the gradients computed
        in the previous forward-backward batch.
        """
        assert self.binded and self.params_initialized and self.optimizer_initialized

        self._params_dirty = True
        if self._update_on_kvstore:
            try:
                _update_params_on_kvstore(self._exec_group.param_arrays,
                                          self._exec_group.grad_arrays,
                                          self._kvstore)
            except:
                _update_params_on_kvstore(self._exec_group.param_arrays,
                                          self._exec_group.grad_arrays,
                                          self._kvstore, param_names=self._exec_group.param_names)
        else:
            _update_params(self._exec_group.param_arrays,
                           self._exec_group.grad_arrays,
                           updater=self._updater,
                           num_device=len(self._context),
                           kvstore=self._kvstore)

    def get_outputs(self, merge_multi_context=True):
        """Get outputs of the previous forward computation.

        Parameters
        ----------
        merge_multi_context : bool
            Default is `True`. In the case when data-parallelism is used, the outputs
            will be collected from multiple devices. A `True` value indicate that we
            should merge the collected results so that they look like from a single
            executor.

        Returns
        -------
        If `merge_multi_context` is `True`, it is like `[out1, out2]`. Otherwise, it
        is like `[[out1_dev1, out1_dev2], [out2_dev1, out2_dev2]]`. All the output
        elements are `NDArray`.
        """
        assert self.binded and self.params_initialized
        return self._exec_group.get_outputs(merge_multi_context=merge_multi_context)

    def get_input_grads(self, merge_multi_context=True):
        """Get the gradients with respect to the inputs of the module.

        Parameters
        ----------
        merge_multi_context : bool
            Default is `True`. In the case when data-parallelism is used, the outputs
            will be collected from multiple devices. A `True` value indicate that we
            should merge the collected results so that they look like from a single
            executor.

        Returns
        -------
        If `merge_multi_context` is `True`, it is like `[grad1, grad2]`. Otherwise, it
        is like `[[grad1_dev1, grad1_dev2], [grad2_dev1, grad2_dev2]]`. All the output
        elements are `NDArray`.
        """
        assert self.binded and self.params_initialized and self.inputs_need_grad
        return self._exec_group.get_input_grads(merge_multi_context=merge_multi_context)

    def get_states(self, merge_multi_context=True):
        """Get states from all devices

        Parameters
        ----------
        merge_multi_context : bool
            Default is `True`. In the case when data-parallelism is used, the states
            will be collected from multiple devices. A `True` value indicate that we
            should merge the collected results so that they look like from a single
            executor.

        Returns
        -------
        If `merge_multi_context` is `True`, it is like `[out1, out2]`. Otherwise, it
        is like `[[out1_dev1, out1_dev2], [out2_dev1, out2_dev2]]`. All the output
        elements are `NDArray`.
        """
        assert self.binded and self.params_initialized
        return self._exec_group.get_states(merge_multi_context=merge_multi_context)

    def set_states(self, states=None, value=None):
        """Set value for states. Only one of states & value can be specified.

        Parameters
        ----------
        states : list of list of NDArrays
            source states arrays formatted like [[state1_dev1, state1_dev2],
            [state2_dev1, state2_dev2]].
        value : number
            a single scalar value for all state arrays.
        """
        assert self.binded and self.params_initialized
        self._exec_group.set_states(states, value)

    def update_metric(self, eval_metric, labels):
        """Evaluate and accumulate evaluation metric on outputs of the last forward computation.

        Parameters
        ----------
        eval_metric : EvalMetric
        labels : list of NDArray
            Typically `data_batch.label`.
        """
        self._exec_group.update_metric(eval_metric, labels)

    def _sync_params_from_devices(self):
        """Synchronize parameters from devices to CPU. This function should be called after
        calling `update` that updates the parameters on the devices, before one can read the
        latest parameters from `self._arg_params` and `self._aux_params`.
        """
        self._exec_group.get_params(self._arg_params, self._aux_params)
        self._params_dirty = False

    def save_optimizer_states(self, fname):
        """Save optimizer (updater) state to file

        Parameters
        ----------
        fname : str
            Path to output states file.
        """
        assert self.optimizer_initialized

        if self._update_on_kvstore:
            self._kvstore.save_optimizer_states(fname)
        else:
            with open(fname, 'wb') as fout:
                fout.write(self._updater.get_states())

    def load_optimizer_states(self, fname):
        """Load optimizer (updater) state from file

        Parameters
        ----------
        fname : str
            Path to input states file.
        """
        assert self.optimizer_initialized

        if self._update_on_kvstore:
            self._kvstore.load_optimizer_states(fname)
        else:
            self._updater.set_states(open(fname, 'rb').read())

    def install_monitor(self, mon):
        """ Install monitor on all executors """
        assert self.binded
        self._exec_group.install_monitor(mon)




class MutableModule(BaseModule):
    """A mutable module is a module that supports variable input data.

    Parameters
    ----------
    symbol : Symbol
    data_names : list of str
    label_names : list of str
    logger : Logger
    context : Context or list of Context
    work_load_list : list of number
    max_data_shapes : list of (name, shape) tuple, designating inputs whose shape vary
    max_label_shapes : list of (name, shape) tuple, designating inputs whose shape vary
    fixed_param_prefix : list of str, indicating fixed parameters
    """
    def __init__(self, symbol, data_names, label_names,
                 logger=logging, context=ctx.cpu(), work_load_list=None,
                 max_data_shapes=None, max_label_shapes=None, fixed_param_prefix=None):
        super(MutableModule, self).__init__(logger=logger)
        self._symbol = symbol
        self._data_names = data_names
        self._label_names = label_names
        self._context = context
        self._work_load_list = work_load_list

        self._curr_module = None
        self._max_data_shapes = max_data_shapes
        self._max_label_shapes = max_label_shapes
        self._fixed_param_prefix = fixed_param_prefix

        fixed_param_names = list()
        if fixed_param_prefix is not None:
            for name in self._symbol.list_arguments():
                for prefix in self._fixed_param_prefix:
                    if prefix in name:
                        fixed_param_names.append(name)
        self._fixed_param_names = fixed_param_names
        self._preload_opt_states = None

    def _reset_bind(self):
        self.binded = False
        self._curr_module = None

    @property
    def data_names(self):
        return self._data_names

    @property
    def output_names(self):
        return self._symbol.list_outputs()

    @property
    def data_shapes(self):
        assert self.binded
        return self._curr_module.data_shapes

    @property
    def label_shapes(self):
        assert self.binded
        return self._curr_module.label_shapes

    @property
    def output_shapes(self):
        assert self.binded
        return self._curr_module.output_shapes

    def get_params(self):
        assert self.binded and self.params_initialized
        return self._curr_module.get_params()

    def init_params(self, initializer=Uniform(0.01), arg_params=None, aux_params=None,
                    allow_missing=False, force_init=False, allow_extra=False):
        if self.params_initialized and not force_init:
            return
        assert self.binded, 'call bind before initializing the parameters'
        self._curr_module.init_params(initializer=initializer, arg_params=arg_params,
                                      aux_params=aux_params, allow_missing=allow_missing,
                                      force_init=force_init)
        self.params_initialized = True

    def bind(self, data_shapes, label_shapes=None, for_training=True,
             inputs_need_grad=False, force_rebind=False, shared_module=None, grad_req='write'):
        # in case we already initialized params, keep it
        if self.params_initialized:
            arg_params, aux_params = self.get_params()

        # force rebinding is typically used when one want to switch from
        # training to prediction phase.
        if force_rebind:
            self._reset_bind()

        if self.binded:
            self.logger.warning('Already binded, ignoring bind()')
            return

        assert shared_module is None, 'shared_module for MutableModule is not supported'

        self.for_training = for_training
        self.inputs_need_grad = inputs_need_grad
        self.binded = True

        max_shapes_dict = dict()
        if self._max_data_shapes is not None:
            max_shapes_dict.update(dict(self._max_data_shapes[0]))
        if self._max_label_shapes is not None:
            max_shapes_dict.update(dict(self._max_label_shapes[0]))

        max_data_shapes = list()
        for name, shape in data_shapes[0]:
            if name in max_shapes_dict:
                max_data_shapes.append((name, max_shapes_dict[name]))
            else:
                max_data_shapes.append((name, shape))

        max_label_shapes = list()
        if not label_shapes.count(None) == len(label_shapes):
            for name, shape in label_shapes[0]:
                if name in max_shapes_dict:
                    max_label_shapes.append((name, max_shapes_dict[name]))
                else:
                    max_label_shapes.append((name, shape))

        if len(max_label_shapes) == 0:
            max_label_shapes = None

        module = Module(self._symbol, self._data_names, self._label_names, logger=self.logger,
                        context=self._context, work_load_list=self._work_load_list,
                        fixed_param_names=self._fixed_param_names)
        module.bind([max_data_shapes for _ in range(len(self._context))], [max_label_shapes for _ in range(len(self._context))],
                    for_training, inputs_need_grad, force_rebind=False, shared_module=None)
        self._curr_module = module

        # copy back saved params, if already initialized
        if self.params_initialized:
            self.set_params(arg_params, aux_params)

    def save_checkpoint(self, prefix, epoch, save_optimizer_states=False):
        """Save current progress to checkpoint.
        Use mx.callback.module_checkpoint as epoch_end_callback to save during training.

        Parameters
        ----------
        prefix : str
            The file prefix to checkpoint to
        epoch : int
            The current epoch number
        save_optimizer_states : bool
            Whether to save optimizer states for continue training
        """
        self._curr_module.save_checkpoint(prefix, epoch, save_optimizer_states)

    def init_optimizer(self, kvstore='local', optimizer='sgd',
                       optimizer_params=(('learning_rate', 0.01),), force_init=False):
        assert self.binded and self.params_initialized
        if self.optimizer_initialized and not force_init:
            self.logger.warning('optimizer already initialized, ignoring.')
            return

        self._curr_module._preload_opt_states = self._preload_opt_states
        self._curr_module.init_optimizer(kvstore, optimizer, optimizer_params,
                                         force_init=force_init)
        self.optimizer_initialized = True

    def fit(self, train_data, eval_data=None, eval_metric='acc',
            epoch_end_callback=None, batch_end_callback=None, kvstore='local',
            optimizer='sgd', optimizer_params=(('learning_rate', 0.01),),
            eval_end_callback=None,
            eval_batch_end_callback=None, initializer=Uniform(0.01),
            arg_params=None, aux_params=None, allow_missing=False,
            force_rebind=False, force_init=False, begin_epoch=0, num_epoch=None,
            validation_metric=None, monitor=None, prefix=None, state=None):
        """Train the module parameters.

        Parameters
        ----------
        train_data : DataIter
        eval_data : DataIter
            If not `None`, will be used as validation set and evaluate the performance
            after each epoch.
        eval_metric : str or EvalMetric
            Default `'acc'`. The performance measure used to display during training.
        epoch_end_callback : function or list of function
            Each callback will be called with the current `epoch`, `symbol`, `arg_params`
            and `aux_params`.
        batch_end_callback : function or list of function
            Each callback will be called with a `BatchEndParam`.
        kvstore : str or KVStore
            Default `'local'`.
        optimizer : str or Optimizer
            Default `'sgd'`
        optimizer_params : dict
            Default `(('learning_rate', 0.01),)`. The parameters for the optimizer constructor.
            The default value is not a `dict`, just to avoid pylint warning on dangerous
            default values.
        eval_end_callback : function or list of function
            These will be called at the end of each full evaluation, with the metrics over
            the entire evaluation set.
        eval_batch_end_callback : function or list of function
            These will be called at the end of each minibatch during evaluation
        initializer : Initializer
            Will be called to initialize the module parameters if not already initialized.
        arg_params : dict
            Default `None`, if not `None`, should be existing parameters from a trained
            model or loaded from a checkpoint (previously saved model). In this case,
            the value here will be used to initialize the module parameters, unless they
            are already initialized by the user via a call to `init_params` or `fit`.
            `arg_params` has higher priority to `initializer`.
        aux_params : dict
            Default `None`. Similar to `arg_params`, except for auxiliary states.
        allow_missing : bool
            Default `False`. Indicate whether we allow missing parameters when `arg_params`
            and `aux_params` are not `None`. If this is `True`, then the missing parameters
            will be initialized via the `initializer`.
        force_rebind : bool
            Default `False`. Whether to force rebinding the executors if already binded.
        force_init : bool
            Default `False`. Indicate whether we should force initialization even if the
            parameters are already initialized.
        begin_epoch : int
            Default `0`. Indicate the starting epoch. Usually, if we are resuming from a
            checkpoint saved at a previous training phase at epoch N, then we should specify
            this value as N+1.
        num_epoch : int
            Number of epochs to run training.

        Examples
        --------
        An example of using fit for training::
            >>> #Assume training dataIter and validation dataIter are ready
            >>> mod.fit(train_data=train_dataiter, eval_data=val_dataiter,
                        optimizer_params={'learning_rate':0.01, 'momentum': 0.9},
                        num_epoch=10)
        """
        assert num_epoch is not None, 'please specify number of epochs'

        self.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label,
                  for_training=True, force_rebind=force_rebind)
        if monitor is not None:
            self.install_monitor(monitor)
        self.init_params(initializer=initializer, arg_params=arg_params, aux_params=aux_params,
                         allow_missing=allow_missing, force_init=force_init)
        self.init_optimizer(kvstore=kvstore, optimizer=optimizer,
                            optimizer_params=optimizer_params)
        if state is not None:
            self._curr_module.load_optimizer_states(state)

        if validation_metric is None:
            validation_metric = eval_metric
        if not isinstance(eval_metric, metric.EvalMetric):
            eval_metric = metric.create(eval_metric)

        ################################################################################
        # training loop
        ################################################################################
        for epoch in range(begin_epoch, num_epoch):
            tic = time.time()
            eval_metric.reset()
            for nbatch, data_batch in enumerate(train_data):
                if monitor is not None:
                    monitor.tic()
                self.forward_backward(data_batch)
                self.update()
                self.update_metric(eval_metric, data_batch.label)

                if monitor is not None:
                    monitor.toc_print()

                if batch_end_callback is not None:
                    batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                     eval_metric=eval_metric,
                                                     locals=locals())
                    for callback in _as_list(batch_end_callback):
                        callback(batch_end_params)

            # one epoch of training is finished
            for name, val in eval_metric.get_name_value():
                self.logger.info('Epoch[%d] Train-%s=%f', epoch, name, val)
            toc = time.time()
            self.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc-tic))

            # sync aux params across devices
            arg_params, aux_params = self.get_params()
            self.set_params(arg_params, aux_params)

            if epoch_end_callback is not None:
                for callback in _as_list(epoch_end_callback):
                    callback(epoch, self.symbol, arg_params, aux_params)

            #----------------------------------------
            # evaluation on validation set
            if eval_data:
                res = self.score(eval_data, validation_metric,
                                 score_end_callback=eval_end_callback,
                                 batch_end_callback=eval_batch_end_callback, epoch=epoch)
                #TODO: pull this into default
                for name, val in res:
                    self.logger.info('Epoch[%d] Validation-%s=%f', epoch, name, val)

            # end of 1 epoch, reset the data-iter for another epoch
            train_data.reset()


    def forward(self, data_batch, is_train=None):
        assert self.binded and self.params_initialized

        # get current_shapes
        if self._curr_module.label_shapes is not None:
            current_shapes = [dict(self._curr_module.data_shapes[i] + self._curr_module.label_shapes[i]) for i in range(len(self._context))]
        else:
            current_shapes = [dict(self._curr_module.data_shapes[i]) for i in range(len(self._context))]

        # get input_shapes
        if is_train:
            input_shapes = [dict(data_batch.provide_data[i] + data_batch.provide_label[i]) for i in range(len(self._context))]
        else:
            input_shapes = [dict(data_batch.provide_data[i]) for i in range(len(data_batch.provide_data))]

        # decide if shape changed
        shape_changed = len(current_shapes) != len(input_shapes)
        for pre, cur in zip(current_shapes, input_shapes):
            for k, v in pre.items():
                if v != cur[k]:
                    shape_changed = True

        if shape_changed:
            # self._curr_module.reshape(data_batch.provide_data, data_batch.provide_label)
            module = Module(self._symbol, self._data_names, self._label_names,
                            logger=self.logger, context=[self._context[i] for i in range(len(data_batch.provide_data))],
                            work_load_list=self._work_load_list,
                            fixed_param_names=self._fixed_param_names)
            module.bind(data_batch.provide_data, data_batch.provide_label, self._curr_module.for_training,
                        self._curr_module.inputs_need_grad, force_rebind=False,
                        shared_module=self._curr_module)
            self._curr_module = module

        self._curr_module.forward(data_batch, is_train=is_train)

    def backward(self, out_grads=None):
        assert self.binded and self.params_initialized
        self._curr_module.backward(out_grads=out_grads)

    def update(self):
        assert self.binded and self.params_initialized and self.optimizer_initialized
        self._curr_module.update()

    def get_outputs(self, merge_multi_context=True):
        assert self.binded and self.params_initialized
        return self._curr_module.get_outputs(merge_multi_context=merge_multi_context)
    def get_input_grads(self, merge_multi_context=True):
        assert self.binded and self.params_initialized and self.inputs_need_grad
        return self._curr_module.get_input_grads(merge_multi_context=merge_multi_context)

    def update_metric(self, eval_metric, labels):
        assert self.binded and self.params_initialized
        self._curr_module.update_metric(eval_metric, labels)

    def install_monitor(self, mon):
        """ Install monitor on all executors """
        assert self.binded
        self._curr_module.install_monitor(mon)
