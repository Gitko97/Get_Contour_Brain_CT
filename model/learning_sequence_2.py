import os

from Dicom.Get_Contour_Brain_CT.model.gan import GAN
from Dicom.Get_Contour_Brain_CT.model.generator import Generator
from Dicom.Get_Contour_Brain_CT.model.discriminator import Discriminator
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.datasets import mnist
from random import randint
import tensorflow as tf
from Dicom.Get_Contour_Brain_CT.model.loss import Loss
from Dicom.Get_Contour_Brain_CT.model.dual_cycle_gan_2 import CycleGan


# class GANMonitor(keras.callbacks.Callback):
#     """A callback to generate and save images after each epoch"""
#
#     def __init__(self, num_img=4):
#         self.num_img = num_img
#
#     def on_epoch_end(self, epoch, logs=None):
#         _, ax = plt.subplots(4, 2, figsize=(12, 12))
#         for i, img in enumerate(test_horses.take(self.num_img)):
#             prediction = self.model.gen_G(img)[0].numpy()
#             prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
#             img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)
#
#             ax[i, 0].imshow(img)
#             ax[i, 1].imshow(prediction)
#             ax[i, 0].set_title("Input image")
#             ax[i, 1].set_title("Translated image")
#             ax[i, 0].axis("off")
#             ax[i, 1].axis("off")
#
#             prediction = keras.preprocessing.image.array_to_img(prediction)
#             prediction.save(
#                 "generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch + 1)
#             )
#         plt.show()
#         plt.close()

gen_G = Generator(name="Generator_G", input_shape=(512, 512, 1)).model
gen_F = Generator(name="Generator_F", input_shape=(512, 512, 1)).model
disc_X = Discriminator(name="Discriminator_X", input_shape=(512, 512, 1)).model
disc_Y = Discriminator(name="Discriminator_Y", input_shape=(512, 512, 1)).model
cycle_gan_model = CycleGan(
    generator_G=gen_G, generator_F=gen_F, discriminator_X=disc_X, discriminator_Y=disc_Y
)

# Compile the model
cycle_gan_model.compile(
    gen_G_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    gen_F_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    disc_X_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    disc_Y_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
)
# Callbacks
# plotter = GANMonitor()
# checkpoint_filepath = "./model_checkpoints/cyclegan_checkpoints.{epoch:03d}"
# model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_filepath
# )

# Here we will train the model for just one epoch as each epoch takes around
# 7 minutes on a single P100 backed machine.
cycle_gan_model.fit(
    [np.zeros((10, 512, 512, 1)), np.zeros((10, 512, 512, 1))],
    epochs=1
    # callbacks=[plotter, model_checkpoint_callback],
)
