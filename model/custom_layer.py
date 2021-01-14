from keras.utils import np_utils
from keras.engine.topology import Layer
from keras.engine import InputSpec
from tensorflow.keras import layers
import tensorflow as tf
from keras.layers import *
import numpy as np


class ReflectionPadding2D(layers.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def get_output_shape_for(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')


class Instance_Normalize(layers.Layer):

    def __init__(self, mean=1.0, stddev=0.02, epsilon=1e-5):
        self.mean = mean
        self.stddev = stddev
        self.epsilon = epsilon
        super(Instance_Normalize, self).__init__()

    def build(self, input_shape):
        self.scale = self.add_weight(name='scale',
                                     shape=[input_shape[3]],
                                     initializer=tf.random_normal_initializer(mean=self.mean, stddev=self.stddev))
        self.offset = self.add_weight(name='offset',
                                      shape=[input_shape[3]],
                                      initializer=tf.constant_initializer(0.0))
        super(Instance_Normalize, self).build(input_shape)

    def call(self, input):
        mean, variance = tf.compat.v1.nn.moments(input, axes=[1, 2], keep_dims=True)
        # normalization
        inv = tf.compat.v1.rsqrt(variance + self.epsilon)
        normalized = (input - mean) * inv
        return self.scale * normalized + self.offset


class Res_Block(layers.Layer):

    def __init__(self, pad_type=None, filters_num=64):
        self.pad_type = pad_type
        self.filters = filters_num
        super(Res_Block, self).__init__()

    def build(self, input_shape):
        if self.pad_type is None:
            self.conv1 = self.conv2 = Conv2D(filters=self.filters, kernel_size=3, padding='same',
                                             kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02),
                                             strides=1,
                                             data_format='channels_last')

        elif self.pad_type == 'REFLECT':
            self.conv1 = self.conv2 = Conv2D(filters=self.filters, kernel_size=3, padding='valid',
                                             kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02),
                                             strides=1,
                                             data_format='channels_last')
            self.padding1 = self.padding2 = ReflectionPadding2D(padding=(1, 1))
        self.normalized1 = self.normalized2 = Instance_Normalize()
        self.relu = ReLU()

    def call(self, input):
        if self.pad_type is None:
            conv1 = self.conv1(input)
        elif self.pad_type == 'REFLECT':
            padded1 = self.padding1(input)
            conv1 = self.conv1(padded1)
        normalized1 = self.normalized1(conv1)
        relu = self.relu(normalized1)

        # 3x3 Conv-Batch S1
        if self.pad_type is None:
            conv2 = self.conv2(relu)
        elif self.pad_type == 'REFLECT':
            padded2 = self.padding2(relu)
            conv2 = self.conv2(padded2)
        normalized2 = self.normalized2(conv2)

        output = input + normalized2
        return output

class Lrelu(layers.Layer):

    def __init__(self, leak=0.2):
        self.leak = leak
        super(Lrelu, self).__init__()

    def call(self, input):
        return tf.maximum(input, self.leak * input)