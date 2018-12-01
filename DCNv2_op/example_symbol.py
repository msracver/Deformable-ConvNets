import mx

"""
demo symbol of using modulated deformable convolution
"""
def modulated_deformable_conv(data, name, num_filter, stride, lr_mult=0.1):
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

"""
demo symbol of using modulated deformable RoI pooling
"""
def modulated_deformable_roi_pool(data, rois, spatial_scale, imfeat_dim=256, deform_fc_dim=1024, roi_size=7, trans_std=0.1):
    roi_align = mx.contrib.sym.DeformablePSROIPooling(name='roi_align', 
                        data=data,
                        rois=rois,
                        group_size=1,
                        pooled_size=roi_size, 
                        sample_per_part=2,
                        no_trans=True, 
                        part_size=roi_size, 
                        output_dim=imfeat_dim,
                        spatial_scale=spatial_scale)

    feat_deform = mx.symbol.FullyConnected(name='fc_deform_1', data=roi_align, num_hidden=deform_fc_dim)
    feat_deform = mx.sym.Activation(data=feat_deform, act_type='relu', name='fc_deform_1_relu')

    feat_deform = mx.symbol.FullyConnected(name='fc_deform_2', data=feat_deform, num_hidden=deform_fc_dim)
    feat_deform = mx.sym.Activation(data=feat_deform, act_type='relu', name='fc_deform_2_relu')
    
    feat_deform = mx.symbol.FullyConnected(name='fc_deform_3', data=feat_deform, num_hidden=roi_size * roi_size * 3)
    
    roi_offset = mx.sym.slice_axis(feat_deform, axis=1, begin=0, end=roi_size * roi_size * 2)
    roi_offset = mx.sym.reshape(roi_offset, shape=(-1, 2, roi_size, roi_size))

    roi_mask = mx.sym.slice_axis(feat_deform, axis=1, begin=roi_size * roi_size * 2, end=None)
    roi_mask_sigmoid = mx.sym.Activation(roi_mask, act_type='sigmoid')
    roi_mask_sigmoid = mx.sym.reshape(roi_mask_sigmoid, shape=(-1, 1, roi_size, roi_size))

    deform_roi_pool = mx.contrib.sym.DeformablePSROIPooling(name='deform_roi_pool', 
                        data=data,
                        rois=rois, 
                        trans=roi_offset,
                        group_size=1,
                        pooled_size=roi_size, 
                        sample_per_part=2,
                        no_trans=False, 
                        part_size=roi_size, 
                        output_dim=imfeat_dim,
                        spatial_scale=spatial_scale, 
                        trans_std=trans_std)

    modulated_deform_roi_pool = mx.sym.broadcast_mul(deform_roi_pool, roi_mask_sigmoid)
    return modulated_deform_roi_pool
