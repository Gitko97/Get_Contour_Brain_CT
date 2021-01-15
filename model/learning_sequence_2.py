import os

import keras
import numpy as np
from tensorflow.python.keras.utils import tf_utils

from Dicom.Get_Contour_Brain_CT.model.discriminator import Discriminator
from Dicom.Get_Contour_Brain_CT.model.dual_cycle_gan import CycleGan
from Dicom.Get_Contour_Brain_CT.model.generator import Generator


inputShape = (128, 128, 1)

gen_G = Generator(name="Generator_G", input_shape=inputShape).model
gen_F = Generator(name="Generator_F", input_shape=(128, 128, 1)).model
disc_X = Discriminator(name="Discriminator_X", input_shape=(128, 128, 1)).model
disc_Y = Discriminator(name="Discriminator_Y", input_shape=(128, 128, 1)).model
cycle_gan_model = CycleGan(
    generator_G=gen_G, generator_F=gen_F, discriminator_X=disc_X, discriminator_Y=disc_Y
)

# Compile the model
cycle_gan_model.compile(
    gen_G_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.9),
    gen_F_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.9),
    disc_X_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.9),
    disc_Y_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.9),
)
# Callbacks
# plotter = LossAndErrorPrintingCallback()
if not os.path.exists("./model_checkpoints"):
    os.mkdir("./model_checkpoints")
checkpoint_filepath = "./model_checkpoints/cyclegan_checkpoints.{epoch:03d}"
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath
)

# Here we will train the model for just one epoch as each epoch takes around
# 7 minutes on a single P100 backed machine.
print("Start")
cycle_gan_model.fit(
    [np.zeros((10, 128, 128, 1)), np.zeros((10, 128, 128, 1))],
    batch_size=5,
    epochs=2,
    callbacks=[model_checkpoint_callback]
)
