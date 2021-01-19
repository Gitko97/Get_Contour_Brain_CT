from tensorflow.python.keras import Model
import tensorflow as tf
from custom_layer import *


class Discriminator:
    def __init__(self, resue_layer=None, ndf=64, input_shape=(512, 512, 1), name=None):
        self.ndf = ndf
        self.name = name
        self.input_shape = input_shape
        self.reuse_layer = resue_layer
        self.model = self.buildModel()
        # self.model.compile(loss='binary_crossentropy', optimizer="adam")

    def lossFunction(self):
        return

    def summary(self):
        return self.model.summary()

    def buildModel(self):
        inputs = Input(self.input_shape)
        conv1 = Conv2D(filters=self.ndf, kernel_size=4, padding='same',
                       kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02),
                       strides=2,
                       data_format='channels_last')(inputs)
        lrelu = Lrelu()(conv1)

        if self.reuse_layer is None:
            self.reuse_layer = ReuseLayer()
        else:
            self.reuse_layer = self.reuse_layer

        reuse_layer_result = self.reuse_layer(lrelu)

        conv5 = Conv2D(filters=1, kernel_size=4, padding='same',
                       kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02),
                       strides=1,
                       data_format='channels_last')(reuse_layer_result)

        result = tf.identity(conv5)

        model = Model(inputs, result, name=self.name)
        return model



class ReuseLayer(layers.Layer):

    def __init__(self, ndf=64):
        self.ndf = ndf
        super(ReuseLayer, self).__init__()

    def build(self, input_shape):
        self.conv2 = Conv2D(filters=self.ndf * 2, kernel_size=4, padding='same',
                            kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02),
                            strides=2,
                            data_format='channels_last')
        self.normalized2 = Instance_Normalize()
        self.lrelu2 = Lrelu()

        self.conv3 = Conv2D(filters=self.ndf * 4, kernel_size=4, padding='same',
                            kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02),
                            strides=2,
                            data_format='channels_last')
        self.normalized3 = Instance_Normalize()
        self.lrelu3 = Lrelu()

        self.conv4 = Conv2D(filters=self.ndf * 8, kernel_size=4, padding='same',
                            kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02),
                            strides=2,
                            data_format='channels_last')
        self.normalized4 = Instance_Normalize()
        self.lrelu4 = Lrelu()

    def call(self, input):
        conv2 = self.conv2(input)
        normalized2 = self.normalized2(conv2)
        lrelu2 = self.lrelu2(normalized2)

        conv3 = self.conv3(lrelu2)
        normalized3 = self.normalized3(conv3)
        lrelu3 = self.lrelu3(normalized3)

        conv4 = self.conv4(lrelu3)
        normalized4 = self.normalized4(conv4)
        lrelu4 = self.lrelu4(normalized4)

        return lrelu4
