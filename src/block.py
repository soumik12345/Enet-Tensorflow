from tensorflow.keras.layers import (
    Conv2D, PReLU, add,
    MaxPooling2D, concatenate,
    Conv2DTranspose, UpSampling2D,
    ZeroPadding2D, BatchNormalization,
    Permute, SpatialDropout2D, Activation
)


def initial_block(input_tensor, filters = 16):
    '''Initial Block Defined as per the Enet Paper
    Reference: https://arxiv.org/pdf/1606.02147.pdf
    Params:
        input_tensor    -> Input Tensor
        filters         -> Number of Output Channels
    '''
    conv = Conv2D(filters - 3, kernel_size=(3, 3), strides=(2, 2), padding='same', name='initial_conv_layer')(input_tensor)
    pool = MaxPooling2D(name='initial_max_pooling_layer')(input_tensor)
    concat = concatenate([conv, pool], axis=3, name='initial_concat_layer')
    return concat


def bottleneck_block_encoder(input_tensor, output_channels, scale = 4, assymetric = 0, dilated = 0, downsample = False, dropout = 0.1):
    '''Bottleneck Block for the Encoder sections
    Reference: https://arxiv.org/pdf/1606.02147.pdf
    Params:
        input_tensor    -> Input Tensor
        output_channels -> Number of Output Channels
        scale           -> Internal Filter Scale
        assymetric      -> Assymetric Convolutional Block
        dilated         -> Dilated Convolutional Block
        downsample      -> Downsampling Bottleneck Block
        dropout         -> Dropout Rate
    '''
    
    ### Branch 1
    
    branch_1 = input_tensor
    
    if downsample:
        branch_1 = MaxPooling2D()(branch_1)
        
        branch_1 = Permute((1, 3, 2))(branch_1)
        
        padded_feature_maps = output_channels - input_tensor.get_shape().as_list()[3]
        vertical_padding = (0, 0)
        horizontal_padding = (0, padded_feature_maps)
        branch_1 = ZeroPadding2D(padding = (vertical_padding, horizontal_padding))(branch_1)
        
        branch_1 = Permute((1, 3, 2))(branch_1)
    
    ### Branch 2
    
    branch_2 = input_tensor
    n_filters = output_channels // scale
    
    # 1x1 or 2x2 Conv Block
    stride = 2 if downsample else 1
    branch_2 = Conv2D(
        n_filters,
        kernel_size = (stride, stride),
        strides = (stride, stride),
        use_bias = False
    )(branch_2)
    branch_2 = BatchNormalization(momentum = 0.1)(branch_2)
    branch_2 = PReLU(shared_axes=[1, 2])(branch_2)

    # Middle Convolutional Block
    if not assymetric and not dilated:
        branch_2 = Conv2D(n_filters, (3, 3), padding = 'same')(branch_2)
    elif assymetric:
        branch_2 = Conv2D(n_filters, (1, assymetric), padding = 'same', use_bias = False)(branch_2)
        branch_2 = Conv2D(n_filters, (assymetric, 1), padding = 'same')(branch_2)
    elif dilated:
        branch_2 = Conv2D(n_filters, (3, 3), padding = 'same', dilation_rate=(dilated, dilated))(branch_2)
    branch_2 = BatchNormalization(momentum = 0.1)(branch_2)
    branch_2 = PReLU(shared_axes=[1, 2])(branch_2)

    # 1x1 Conv Block
    branch_2 = Conv2D(output_channels, (1, 1), use_bias = False)(branch_2)
    branch_2 = BatchNormalization(momentum = 0.1)(branch_2)
    branch_2 = SpatialDropout2D(dropout)(branch_2)

    # Branch_1 + Branch_2
    merged = add([branch_1, branch_2])
    merged = PReLU(shared_axes=[1, 2])(merged)

    return merged


def bottleneck_block_decoder(input_tensor, output_channels, scale = 4, upsample = False, reverse_module = False):
    '''Bottleneck Block for the Decoder sections
    Reference: https://arxiv.org/pdf/1606.02147.pdf
    Params:
        input_tensor    -> Input Tensor
        output_channels -> Number of Output Channels
        scale           -> Internal Filter Scale
        upsample        -> Upsampling Bottleneck Block
        reverse_module  -> Reverse Module
    '''
    
    ### Branch 1

    branch_1 = input_tensor
    n_filters = output_channels // scale

    # 1x1 conv block
    branch_1 = Conv2D(n_filters, (1, 1), use_bias=False)(branch_1)
    branch_1 = BatchNormalization(momentum = 0.1)(branch_1)
    branch_1 = Activation('relu')(branch_1)

    # Mid Convolutional Block
    if upsample:
        branch_1 = Conv2DTranspose(filters=n_filters, kernel_size=(3, 3), strides=(2, 2), padding='same')(branch_1)
    else:
        branch_1 = Conv2D(n_filters, (3, 3), padding='same', use_bias=True)(branch_1)
    branch_1 = BatchNormalization(momentum=0.1)(branch_1)
    branch_1 = Activation('relu')(branch_1)

    # 1x1 conv block
    branch_1 = Conv2D(output_channels, (1, 1), padding='same', use_bias=False)(branch_1)

    ### Branch 2
    branch_2 = input_tensor

    if input_tensor.get_shape()[-1] != output_channels or upsample:
        branch_2 = Conv2D(output_channels, (1, 1), padding='same', use_bias=False)(branch_2)
        branch_2 = BatchNormalization(momentum=0.1)(branch_2)
        if upsample and reverse_module is True:
            branch_2 = UpSampling2D(size=(2, 2))(branch_2)
    
    ### Branch Merger
    if upsample and reverse_module is False:
        decoder = branch_1
    else:
        branch_1 = BatchNormalization(momentum=0.1)(branch_1)
        decoder = add([branch_1, branch_2])
        decoder = Activation('relu')(decoder)
    
    return decoder