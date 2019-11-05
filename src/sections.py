from .block import *


def section_1(input_tensor):
    '''Section 1 of architecture as per the Enet Paper
    Reference: https://arxiv.org/pdf/1606.02147.pdf
    Params:
        input_tensor    -> Input Tensor
    '''
    x = initial_block(input_tensor, 16)
    x = bottleneck_block_encoder(x, 64, downsample=True, dropout=0.01)
    for _ in range(4):
        x = bottleneck_block_encoder(x, 64, downsample=False, dropout=0.01)
    return x


def section_2(input_tensor):
    '''Section 2 of architecture as per the Enet Paper
    Reference: https://arxiv.org/pdf/1606.02147.pdf
    Params:
        input_tensor    -> Input Tensor
    '''
    x = bottleneck_block_encoder(input_tensor, 128, downsample=True)
    x = bottleneck_block_encoder(x, 128)
    x = bottleneck_block_encoder(x, 128, dilated=2)
    x = bottleneck_block_encoder(x, 128, assymetric=5)
    x = bottleneck_block_encoder(x, 128, dilated=4)
    x = bottleneck_block_encoder(x, 128)
    x = bottleneck_block_encoder(x, 128, dilated=8)
    x = bottleneck_block_encoder(x, 128, assymetric=5)
    x = bottleneck_block_encoder(x, 128, dilated=16)
    return x


def section_3(input_tensor):
    '''Section 3 of architecture as per the Enet Paper
    Reference: https://arxiv.org/pdf/1606.02147.pdf
    Params:
        input_tensor    -> Input Tensor
    '''
    x = bottleneck_block_encoder(input_tensor, 128)
    x = bottleneck_block_encoder(x, 128, dilated=2)
    x = bottleneck_block_encoder(x, 128, assymetric=5)
    x = bottleneck_block_encoder(x, 128, dilated=4)
    x = bottleneck_block_encoder(x, 128)
    x = bottleneck_block_encoder(x, 128, dilated=8)
    x = bottleneck_block_encoder(x, 128, assymetric=5)
    x = bottleneck_block_encoder(x, 128, dilated=16)
    return x


def section_4(input_tensor):
    '''Section 4 of architecture as per the Enet Paper
    Reference: https://arxiv.org/pdf/1606.02147.pdf
    Params:
        input_tensor    -> Input Tensor
    '''
    x = bottleneck_block_decoder(input_tensor, 64, upsample=True, reverse_module=True)
    x = bottleneck_block_decoder(x, 64)
    x = bottleneck_block_decoder(x, 64)
    return x


def section_5(input_tensor):
    '''Section 5 of architecture as per the Enet Paper
    Reference: https://arxiv.org/pdf/1606.02147.pdf
    Params:
        input_tensor    -> Input Tensor
    '''
    x = bottleneck_block_decoder(input_tensor, 16, upsample=True, reverse_module=True)
    x = bottleneck_block_decoder(x, 16)
    return x