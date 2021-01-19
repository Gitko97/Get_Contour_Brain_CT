import argparse
import sys
import time

import keras
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

sys.path.append('../')
from callback import GANMonitor
from discriminator import Discriminator
from dual_cycle_gan import CycleGan
from generator import Generator
from load_dataset import LoadDataSet

seed = 0
np.random.seed(seed)

parser = argparse.ArgumentParser("Cycle Gan")

parser.add_argument("--d", help="for polestar data path", required=False)
parser.add_argument("--e", help="for train epoch", required=False)
parser.add_argument("--b", help="for train batch size", required=False)

args = parser.parse_args()
inputShape = (512, 512, 1)


def get_options_with_parameter():
    EPOCHS = int(args.e)
    BATCH_SIZE = int(args.b)
    loader = LoadDataSet(args.d, input_shape=inputShape)

    return EPOCHS, BATCH_SIZE, loader


def get_options_with_default():
    EPOCHS = 3
    BATCH_SIZE = 1
    loader = LoadDataSet("/Users/joonhyoungjeon/Downloads/20201221/S156", input_shape=inputShape)

    return EPOCHS, BATCH_SIZE, loader


EPOCHS, BATCH_SIZE, loader = get_options_with_default()
# EPOCHS, BATCH_SIZE, loader = get_options_with_parameter()


# ------Now try to load DATA-------#

# ToDo:Change to Tensorflow Dataset
ct_data, mr_data = loader.change_to_pixel_array()
CT_train, CT_test, MR_train, MR_test = train_test_split(ct_data, mr_data, test_size=0.1)

CT_train = CT_train.reshape(CT_train.shape + (1,))
CT_test = CT_test.reshape(CT_test.shape + (1,))
MR_train = MR_train.reshape(MR_train.shape + (1,))
MR_test = MR_test.reshape(MR_test.shape + (1,))

print("The Number of CT Train Data is {}".format(np.shape(CT_train)[0]))
print("The Number of MR Train Data is {}".format(np.shape(MR_train)[0]))
print("The Number of CT Test Data is {}".format(np.shape(CT_test)[0]))
print("The Number of MR Test Data is {}".format(np.shape(MR_test)[0]))

train_dataset_CT = tf.data.Dataset.from_tensor_slices(CT_train).cache().shuffle(
    512).batch(BATCH_SIZE)
train_dataset_MR = tf.data.Dataset.from_tensor_slices(MR_train).cache().shuffle(
    512).batch(BATCH_SIZE)

# ------Make Cycle Gan(only for unpaired data)-------#


gen_G = Generator(name="Generator_G", input_shape=inputShape).model
gen_F = Generator(name="Generator_F", input_shape=inputShape).model
disc_X = Discriminator(name="Discriminator_X", input_shape=inputShape).model
disc_Y = Discriminator(name="Discriminator_Y", input_shape=inputShape).model
generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.9)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.9)
discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.9)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.9)
cycle_gan_model = CycleGan(
    generator_G=gen_G, generator_F=gen_F, discriminator_X=disc_X, discriminator_Y=disc_Y
)

# Compile the model
cycle_gan_model.compile(
    gen_G_optimizer=generator_g_optimizer,
    gen_F_optimizer=generator_f_optimizer,
    disc_X_optimizer=discriminator_x_optimizer,
    disc_Y_optimizer=discriminator_y_optimizer,
)

# ------For train callback method-------#
imgae_callback = GANMonitor(gen_G, test_dataset=CT_test)
checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(generator_g=gen_G,
                           generator_f=gen_F,
                           discriminator_x=disc_X,
                           discriminator_y=disc_Y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

K = tf.keras.backend


def train():
    for epoch in range(EPOCHS):
        start = time.time()

        n = 0
        for image_x, image_y in tf.data.Dataset.zip((train_dataset_CT, train_dataset_MR)):
            history = cycle_gan_model.train_step(image_x, image_y)
            # K.clear_session() # this method can be solution for memory leak
            if n % 1 == 0:
                print("{} Epochs / {} batch History :".format(epoch+1, n))
                print(history)
            n += 1

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            imgae_callback(epoch)
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                           time.time() - start))



train()
