import argparse
import os
import sys
import time

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
# -*- coding: utf-8 -*-
sys.path.append('../')
from callback import GANMonitor, MyTensorBoard
from discriminator import Discriminator
from dual_cycle_gan import CycleGan
from generator import Generator
from load_dataset import LoadDataSet

# tf.debugging.set_log_device_placement(True)  # print gpu allocate
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


if args.d is None:
    EPOCHS, BATCH_SIZE, loader = get_options_with_default()
else:
    EPOCHS, BATCH_SIZE, loader = get_options_with_parameter()

strategy = tf.distribute.MirroredStrategy()

print("Number of devices: {}".format(strategy.num_replicas_in_sync))
global_batch_size = (BATCH_SIZE *
                     strategy.num_replicas_in_sync)

# ------Now try to load DATA-------#

CT_train, MR_train = loader.change_to_pixel_array()

CT_train = CT_train.reshape(CT_train.shape + (1,))
MR_train = MR_train.reshape(MR_train.shape + (1,))

print("The Number of CT Train Data is {}".format(np.shape(CT_train)[0]))
print("The Number of MR Train Data is {}".format(np.shape(MR_train)[0]))

with strategy.scope():
    train_dataset = tf.data.Dataset.from_tensor_slices((CT_train, MR_train)).shuffle(len(CT_train)).batch(
        global_batch_size,
        drop_remainder=True)
    # Remain Batch Data removed!
    dist_dataset = strategy.experimental_distribute_dataset(train_dataset)

# Generator(name="TEST G", input_shape=inputShape).print_vars()
# Discriminator(name="TEST D", input_shape=inputShape).print_vars()
# s = input()
# ------Make Cycle Gan(only for unpaired data)-------#
with strategy.scope():
    gen_G = Generator(name="Generator_G", input_shape=inputShape).model
    gen_F = Generator(name="Generator_F", input_shape=inputShape).model
    disc_X = Discriminator(name="Discriminator_X", input_shape=inputShape).model
    disc_Y = Discriminator(name="Discriminator_Y", input_shape=inputShape).model

    generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.9)
    generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.9)
    discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.9)
    discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.9)

    cycle_gan_model = CycleGan(
        generator_G=gen_G, generator_F=gen_F, discriminator_X=disc_X, discriminator_Y=disc_Y,
        batch_size=global_batch_size
    )

    # Compile the model
    cycle_gan_model.compile(
        gen_G_optimizer=generator_g_optimizer,
        gen_F_optimizer=generator_f_optimizer,
        disc_X_optimizer=discriminator_x_optimizer,
        disc_Y_optimizer=discriminator_y_optimizer,
    )




# ------For train callback method-------#
imgae_callback = GANMonitor(gen_G, dir_path="./generate_img/", test_dataset=CT_train)

if not os.path.exists("./checkpoints"):
    os.mkdir("./checkpoints")
checkpoint_path = "./checkpoints"
with strategy.scope():
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

loss_log_tensorboard = MyTensorBoard("./logs_tensorboard/")

with strategy.scope():
    @tf.function
    def call_train(image_x, image_y, strategy):
        history = cycle_gan_model.train_step(image_x, image_y, strategy)

        return history


    def train():
        for epoch in range(EPOCHS):
            start = time.time()

            n = 0 + global_batch_size
            for image_x, image_y in dist_dataset:
                history = call_train(image_x, image_y, strategy)
                # K.clear_session() # this method can be solution for memory leak
                print("{} Epochs / {} batch History :".format(epoch + 1, n))
                template = 'G_loss: {}, F_loss: {}, D_X_loss: {}, D_Y_loss Loss: {}'
                print(template.format(history['G_loss'],
                                      history['F_loss'],
                                      history['D_X_loss'],
                                      history['D_Y_loss']))
                n += global_batch_size

            if (epoch + 1) % 5 == 0:
                ckpt_save_path = ckpt_manager.save()
                imgae_callback(epoch)
                print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                    ckpt_save_path))

            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                               time.time() - start))
            loss_log_tensorboard.write_log(history, epoch)

print("Start Learning")
train()
