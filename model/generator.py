from tensorflow import keras
from tensorflow.python.keras import Model
from tensorflow.python.keras.optimizer_v2.adam import Adam
from keras.activations import tanh
from Dicom.Get_Contour_Brain_CT.model.custom_layer import *


class Generator:
    def __init__(self, input_shape=(512,512,1), name=None):
        self.ngf = 64
        self.input_shape = input_shape
        self.name = name
        self.model = self.buildModel()
        # self.model.compile(loss='binary_crossentropy', optimizer="adam")

    def lossFunction(self):
        return

    def buildModel(self):
        # (N, H, W, C) -> (N, H, W, 64)
        inputs = Input(self.input_shape)
        padded_inputs = ReflectionPadding2D(padding=(3, 3))(inputs)
        conv1 = Conv2D(filters=self.ngf, kernel_size=7, padding='valid', kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02),
                       strides=1,
                       data_format='channels_last')(padded_inputs)
        normalized = Instance_Normalize()(conv1)
        relu = ReLU()(normalized)

        conv2 = Conv2D(filters=self.ngf * 2, kernel_size=3, padding='same', kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02),
                       strides=2,
                       data_format='channels_last')(relu)
        normalized = Instance_Normalize()(conv2)
        relu = ReLU()(normalized)

        conv3 = Conv2D(filters=self.ngf * 4, kernel_size=3, padding='same',
                       kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02),
                       strides=2,
                       data_format='channels_last')(relu)
        normalized = Instance_Normalize()(conv3)
        relu = ReLU()(normalized)

        res_out = relu
        if (self.input_shape[0] <= 128) and (self.input_shape[1] <= 128):
            # use 6 residual blocks for 128x128 images
            for idx in range(1, 6 + 1):
                res_out = Res_Block(filters_num=np.shape(res_out)[3])(res_out)
        else:
            # use 9 blocks for higher resolution
            for idx in range(1, 9 + 1):
                res_out = Res_Block(filters_num=np.shape(res_out)[3])(res_out)

        conv4 = Conv2DTranspose(filters=2 * self.ngf, kernel_size=3, strides=2, padding="same")(res_out)
        norm = Instance_Normalize()(conv4)
        relu = ReLU()(norm)

        conv5 = Conv2DTranspose(filters=self.ngf, kernel_size=3, strides=2, padding="same")(relu)
        norm = Instance_Normalize()(conv5)
        relu = ReLU()(norm)

        padded_inputs = ReflectionPadding2D(padding=(3, 3))(relu)
        norm = Conv2D(filters=self.input_shape[2], kernel_size=7, padding='valid', kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02),
                       strides=1,
                       data_format='channels_last')(padded_inputs)
        result = tanh(norm)
        model = Model(inputs, result, name=self.name)

        return model

    def trainModel(self, input_data):
        return self.model.predict(input_data)

    def summary(self):
        return self.model.summary()


