#Developed by Mohsen Avdolahzadeh Kondori
import skimage.io as io
import skimage.transform as trans
import matplotlib.pyplot as plt
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, add
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.utils.vis_utils import plot_model
import numpy as np 
import os
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

def MSPoreConv_2D(x,
              filters,
              num_row,
              num_col,
              padding = 'same',
              strides=(1, 1),
              activation = 'relu',
              name = None):

    x = Conv2D(filters,
               (num_row, num_col),
               strides  = strides,
               padding  = padding,
               use_bias = False)(x)
    x = BatchNormalization(axis = 3, scale = False)(x)

    if(activation == None):
        return x

    x = Activation(activation, name = name)(x)

    return x


def MSPoreBlock(inputFilter,
                inpConv):

    W = 1.67 * inputFilter

    shortcut = inpConv

    shortcut = MSPoreConv_2D(shortcut,
                         int(W * 0.167) + int(W * 0.333) + int(W * 0.5),
                         1,
                         1,
                         activation = None,
                         padding    = 'same')

    conv3x3  = MSPoreConv_2D(inpConv,
                         int(W * 0.167),
                         3,
                         3,
                        activation = 'relu',
                         padding = 'same')

    conv5x5  = MSPoreConv_2D(conv3x3,
                         int(W * 0.333),
                         3,
                         3,
                        activation='relu',
                         padding='same')

    conv7x7  = MSPoreConv_2D(conv5x5,
                         int(W * 0.5),
                         3,
                         3,
                        activation='relu',
                         padding='same')

    out = concatenate([conv3x3,
                       conv5x5,
                       conv7x7],
                      axis = 3)
    out = BatchNormalization(axis = )(out)

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    return out


def MSPoreSkipSequence(filters,
                       length,
                       inp):

    shortcut = inp
    shortcut = MSPoreConv_2D(shortcut,
                         filters,
                         1,
                         1,
                         activation = None,
                         padding = 'same')

    out = MSPoreConv_2D(inp,
                        filters,
                        3,
                        3,
                        activation = 'relu',
                        padding    = 'same')

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis = 3)(out)

#MSPoreSkip Blocks
    for i in range(length-1):

        shortcut = out
        shortcut = MSPoreConv_2D(shortcut,
                             filters,
                             1,
                             1,
                             activation=None,
                             padding = 'same')

        out = MSPoreConv_2D(out,
                        filters,
                        3,
                        3,
                        activation='relu',
                        padding='same')

        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization(axis = 3)(out)

    return out


def MSPoreUNet(height,
               width,
               n_channels):

    baseFilter = 32
    raiseFilter = 1
    inputs = Input((height,
                    width,
                    n_channels))
    
#Convolution Block 1
    MSPoreBlock1 = MSPoreBlock(baseFilter * raiseFilter, inputs)
    pool1      = MaxPooling2D(pool_size=(2, 2))(MSPoreBlock1)
    mresblock1 = MSPoreSkipSequence(baseFilter * raiseFilter,
                                    4,
                                    MSPoreBlock1)

#Convolution Block 2
    raiseFilter = 2
    MSPoreBlock2 = MSPoreBlock(baseFilter * raiseFilter, pool1)
    pool2      = MaxPooling2D(pool_size=(2, 2))(MSPoreBlock2)
    MSPoreBlock2 = MSPoreSkipSequence(baseFilter * raiseFilter,
                                      3,
                                      MSPoreBlock2)

#Convolution Block 3
    raiseFilter = 4
    MSPoreBlock3 = MSPoreBlock(baseFilter * raiseFilter, pool2)
    pool3      = MaxPooling2D(pool_size=(2, 2))(MSPoreBlock3)
    MSPoreBlock3 = MSPoreSkipSequence(baseFilter * raiseFilter,
                                      2,
                                      MSPoreBlock3)
    
#Convolution Block 4
    raiseFilter = 8
    MSPoreBlock4 = MSPoreBlock(baseFilter * raiseFilter, pool3)
    pool4      = MaxPooling2D(pool_size=(2, 2))(MSPoreBlock4)
    MSPoreBlock4 = MSPoreSkipSequence(baseFilter * 8,
                                    1,
                                    MSPoreBlock4)
    
#Convolution Block 5
    raiseFilter = 16
    MSPoreBlock5 = MSPoreBlock(baseFilter * raiseFilter, pool4)

    up6 = concatenate([Conv2DTranspose(baseFilter * raiseFilter / 2,
                                       (2, 2),
                                       strides=(2, 2),
                                       padding='same')(MSPoreBlock5),
                       mresblock4],
                      axis=3)
    MSPoreBlock6 = MSPoreBlock(baseFilter * raiseFilter / 2, up6)

    up7 = concatenate([Conv2DTranspose(baseFilter * 4,
                                       (2, 2),
                                       strides=(2, 2),
                                       padding='same')(MSPoreBlock6),
                       MSPoreBlock3],
                      axis = 3)
    raiseFilter = 4
    MSPoreBlock7 = MSPoreBlock(baseFilter * raiseFilter, up7)

    up8 = concatenate([Conv2DTranspose(baseFilter * raiseFilter / 2,
                                       (2, 2),
                                       strides = (2, 2),
                                       padding='same')(MSPoreBlock7),
                       MSPoreBlock2],
                      axis = 3)
    MSPoreBlock8 = MSPoreBlock(baseFilter * raiseFilter / 2, up8)

    up9 = concatenate([Conv2DTranspose(baseFilter,
                                       (2, 2),
                                       strides=(2, 2),
                                       padding='same')(MSPoreBlock8),
                       MSPoreBlock1],
                      axis = 3)
    MSPoreBlock9 = MSPoreBlock(baseFilter, up9)

    conv10 = MSPoreConv_2D(MSPoreBlock9,
                           1,
                           1,
                           1,
                           activation = 'sigmoid')
    
    model = Model(inputs=[inputs],
                  outputs=[conv10])

    model.compile(optimizer = Adam(lr = 1e-4),
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])
    
    return model
