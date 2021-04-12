import os
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import tensorflow_addons as tfa
import tensorflow as tf

def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding = 'same', 
               kernel_initializer = tf.random_normal_initializer(0, np.sqrt(4.0/num_filters)))(inputs)
    x = ReLU()(x)
    x = Conv2D(num_filters, 3, padding = 'same', 
               kernel_initializer = tf.random_normal_initializer(0, np.sqrt(2.0/num_filters)))(x)
    x = ReLU()(x)
    return x

def enc_conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding = 'same', 
               kernel_initializer = tf.random_normal_initializer(0, np.sqrt(4.0/num_filters)))(inputs)
    x = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True)(x)
#    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(num_filters, 3, padding = 'same', 
               kernel_initializer = tf.random_normal_initializer(0, np.sqrt(2.0/num_filters)))(x)
    x = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True)(x)
#    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def dec_conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding = 'same', 
               kernel_initializer = tf.random_normal_initializer(0, np.sqrt(1.0/num_filters)))(inputs)
    x = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True)(x)
#    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(num_filters, 3, padding = 'same', 
               kernel_initializer = tf.random_normal_initializer(0, np.sqrt(2.0/num_filters)))(x)
#    x = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, 
#                     beta_initializer="glorot_normal", 
#                     gamma_initializer="glorot_normal")(x)
    x = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True)(x)
#    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

# 0.1    0.1    0.2    0.3    0.3    0.2    0.2    0.1    0.1
def reversed_enumerate(seq, start=None):
    if start is None:
        start = len(seq) - 1
    n = start
    for el in reversed(seq):
        yield n, el
        n -= 1

def unet(input_shape, num_filters, num_classes, dropout_ratio, num_layers = 4):
    inputs = Input(input_shape)
    x = inputs
    down_layers = []
    
    for i in range(num_layers):
        x = enc_conv_block(x, num_filters)
        down_layers.append(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(dropout_ratio)(x)
        num_filters *= 2

    x = conv_block(x, num_filters)
    
    for i, layer in reversed_enumerate(down_layers):
        num_filters //= 2
        x = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding="same")(x)
        x = concatenate([x, layer])
        x = dec_conv_block(x, num_filters)
        if i > 0: # Don't dropout on the last layer
            x = Dropout(dropout_ratio)(x)
    outputs = Conv2D(num_classes, (1,1), activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model
