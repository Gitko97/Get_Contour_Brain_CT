from datetime import datetime
import os

import cv2
import tensorflow as tf
import keras
import numpy as np
from image_utils.preprocessor import Image_PreProcessor


def save_check_points():
    if not os.path.exists("./model_checkpoints"):
        os.mkdir("./model_checkpoints")
    checkpoint_filepath = "./model_checkpoints/cyclegan_checkpoints.{epoch:03d}"
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath, save_freq=1, save_weights_only=True,
    )
    return model_checkpoint_callback


class MyTensorBoard:
    def __init__(self, log_dir="./"):
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        log_dir = os.path.join(log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.train_summary_writer = tf.summary.create_file_writer(log_dir)

    def write_log(self, history, epoch):
        with self.train_summary_writer.as_default():
            tf.summary.scalar('G_loss', history['G_loss'].numpy(), step=epoch)
            tf.summary.scalar('F_loss', history['F_loss'].numpy(), step=epoch)
            tf.summary.scalar('D_X_loss', history['D_X_loss'].numpy(), step=epoch)
            tf.summary.scalar('D_Y_loss', history['D_Y_loss'].numpy(), step=epoch)


class GANMonitor(keras.callbacks.Callback):
    """A callback to generate and save images after each epoch"""

    def __init__(self, gen_G, test_dataset, dir_path="./", num_img=4):
        self.num_img = num_img
        self.test_dataset = test_dataset
        self.gen_G = gen_G
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        self.sample_directory = dir_path
        self.test_dataset_num = np.shape(test_dataset)[0]
        self.arrayToPng = Image_PreProcessor().unNormalizeToPNG

    def __call__(self, epoch, logs=None):
        for i, index in enumerate(np.random.choice(self.test_dataset_num, self.num_img, replace=False)):
            img = self.test_dataset[index].reshape((1,) + self.test_dataset[index].shape)
            prediction = self.gen_G(tf.convert_to_tensor(img))[0].numpy()
            prediction = self.arrayToPng(prediction,min_bound=-10, max_bound=2000,pixel_mean=0.25)
            img = self.arrayToPng(img[0],min_bound=-1000, max_bound=4000, pixel_mean=0.25)
            prediction = np.array(keras.preprocessing.image.array_to_img(prediction))
            img = np.array(keras.preprocessing.image.array_to_img(img))
            result = cv2.hconcat([img, prediction])
            cv2.imwrite(filename=self.sample_directory + "generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch + 1)
                        , img=result)

