from tensorflow.keras.layers import Input, Conv2DTranspose
from tensorflow.keras.models import Model
from .sections import *


def Enet(input_shape = (512, 512, 3), output_channels = 3):
    '''The Enet architecture as per the Enet Paper
    Reference: https://arxiv.org/pdf/1606.02147.pdf
    Params:
        input_shape     -> Input Shape
        output_channels -> Number of channels in the output mask
    '''
    input_tensor = Input(shape = input_shape, name = 'Input_Layer')
    
    x = section_1(input_tensor)
    x = section_2(x)
    x = section_3(x)
    x = section_4(x)
    x = section_5(x)
    
    output_tennsor = Conv2DTranspose(filters=output_channels, kernel_size=(2, 2), strides=(2, 2), padding='same')(x)
    
    model = Model(input_tensor, output_tennsor)
    return model