from datetime import datetime
import os

import cv2
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np


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

    def __call__(self, epoch, logs=None):
        _, ax = plt.subplots(4, 2, figsize=(12, 12))
        for i, img in enumerate(self.test_dataset.take(self.num_img)):
            prediction = self.gen_G(tf.convert_to_tensor(img))[0].numpy()
            prediction = (prediction * 3000 + (-1000)).astype(np.uint8)
            img = (img[0] * 3000 + (-1000)).numpy().astype(np.uint8)

            ax[i, 0].imshow(img)
            ax[i, 1].imshow(prediction)
            ax[i, 0].set_title("Input image")
            ax[i, 1].set_title("Translated image")
            ax[i, 0].axis("off")
            ax[i, 1].axis("off")

            prediction = np.array(keras.preprocessing.image.array_to_img(prediction, scale=True))
            img = np.array(keras.preprocessing.image.array_to_img(img, scale=True))
            result = cv2.hconcat([img, prediction])
            cv2.imwrite(filename=self.sample_directory + "generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch + 1)
                        ,img=result)
