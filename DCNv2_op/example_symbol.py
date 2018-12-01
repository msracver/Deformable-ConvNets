"""
demo symbol of using modulated deformable convolution
"""
def modulated_deformable_conv(data, name, num_filter, stride, lr_mult=1):
    weight_var = mx.sym.Variable(name=name+'_conv2_offset_weight', init=mx.init.Zero(), lr_mult=lr_mult)
    bias_var = mx.sym.Variable(name=name+'_conv2_offset_bias', init=mx.init.Zero(), lr_mult=lr_mult)
    conv2_offset = mx.symbol.Convolution(name=name + '_conv2_offset', data=data, num_filter=27,
                       pad=(1, 1), kernel=(3, 3), stride=stride, weight=weight_var, bias=bias_var, lr_mult=lr_mult)
    conv2_offset_t = mx.sym.slice_axis(conv2_offset, axis=1, begin=0, end=18)
    conv2_mask =  mx.sym.slice_axis(conv2_offset, axis=1, begin=18, end=None)
    conv2_mask = 2 * mx.sym.Activation(conv2_mask, act_type='sigmoid')

    conv2 = mx.contrib.symbol.ModulatedDeformableConvolution(name=name + '_conv2', data=act1, offset=conv2_offset_t, mask=conv2_mask,
                       num_filter=num_filter, pad=(1, 1), kernel=(3, 3), stride=stride, 
                       num_deformable_group=1, no_bias=True)
    return conv2
