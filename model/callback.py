import os
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

class GANMonitor(keras.callbacks.Callback):
    """A callback to generate and save images after each epoch"""

    def __init__(self, gen_G, test_ct_data, num_img=4):
        self.num_img = num_img
        self.test_ct = test_ct_data
        self.gen_G = gen_G

    def __call__(self, epoch, logs=None):
        _, ax = plt.subplots(4, 2, figsize=(12, 12))
        for i, img in enumerate(self.test_ct.take(self.num_img)):
            prediction = self.model.gen_G(tf.convert_to_tensor(img))[0].numpy()
            prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
            img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)

            ax[i, 0].imshow(img)
            ax[i, 1].imshow(prediction)
            ax[i, 0].set_title("Input image")
            ax[i, 1].set_title("Translated image")
            ax[i, 0].axis("off")
            ax[i, 1].axis("off")

            prediction = keras.preprocessing.image.array_to_img(prediction)
            prediction.save(
                "generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch + 1)
            )
        plt.show()
        plt.close()
