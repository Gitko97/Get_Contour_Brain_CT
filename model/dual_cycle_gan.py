import keras
import tensorflow as tf
import numpy as np
from loss import Loss
from tensorflow.python.distribute.mirrored_strategy import MirroredStrategy
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Reshape


class CycleGan:
    def __init__(
            self,
            generator_G,
            generator_F,
            discriminator_X,
            discriminator_Y,
            batch_size,
            lambda_cycle=10.0,
    ):
        self.gen_G = generator_G
        self.gen_F = generator_F
        self.disc_X = discriminator_X
        self.disc_Y = discriminator_Y
        self.lambda_cycle = lambda_cycle
        self.loss_function = Loss()
        self.number = 0
        self.batch_size = batch_size

    def compile(
            self,
            gen_G_optimizer,
            gen_F_optimizer,
            disc_X_optimizer,
            disc_Y_optimizer
    ):
        self.gen_G_optimizer = gen_G_optimizer
        self.gen_F_optimizer = gen_F_optimizer
        self.disc_X_optimizer = disc_X_optimizer
        self.disc_Y_optimizer = disc_Y_optimizer
        self.generator_loss_fn = self.loss_function.generator_loss
        self.discriminator_loss_fn = self.loss_function.discriminator_loss
        self.cycle_loss_fn = self.loss_function.cycle_consistency_loss
        self.gradient_difference_loss = self.loss_function.gradient_difference_loss
        self.perceptual_loss_fn = self.loss_function.perceptual_loss_fn
        self.ssim_loss_fn = self.loss_function.ssim_loss_fn

    @tf.function  # change to graph mode
    def train_step(self, ct, mr, strategy: MirroredStrategy):
        def calculate(ct, mr, batch_size):
            with tf.GradientTape(persistent=True) as tape:
                # CT to fake mri
                fake_mr = self.gen_G(ct, training=True)
                # MRI to fake CT
                fake_ct = self.gen_F(mr, training=True)

                # Cycle (CT to fake MRI to fake CT)
                cycled_x = self.gen_F(fake_mr, training=True)
                # Cycle (MRI to fake CT to fake MRI)
                cycled_y = self.gen_G(fake_ct, training=True)

                # Discriminator output
                disc_real_x = self.disc_X(ct, training=True)
                disc_fake_x = self.disc_X(fake_ct, training=True)

                disc_real_y = self.disc_Y(mr, training=True)
                disc_fake_y = self.disc_Y(fake_mr, training=True)

                # Generator adverserial loss
                gen_G_loss = self.generator_loss_fn(disc_fake_y)
                gen_F_loss = self.generator_loss_fn(disc_fake_x)

                # Divide By batch_size because calculate with MultiGPU
                gen_G_loss = tf.reduce_sum(gen_G_loss) * (1.0 / batch_size)
                gen_F_loss = tf.reduce_sum(gen_F_loss) * (1.0 / batch_size)

                # Generator cycle loss
                cycle_loss_G = self.cycle_loss_fn(mr, cycled_y, self.lambda_cycle)
                cycle_loss_F = self.cycle_loss_fn(ct, cycled_x, self.lambda_cycle)

                # Divide By batch_size because calculate with MultiGPU
                cycle_loss_G = tf.reduce_sum(cycle_loss_G) * (1.0 / batch_size)
                cycle_loss_F = tf.reduce_sum(cycle_loss_F) * (1.0 / batch_size)

                # Gradient loss
                gradient_difference_loss_G = self.gradient_difference_loss(fake_mr, mr)
                gradient_difference_loss_F = self.gradient_difference_loss(fake_ct, ct)

                # Divide By batch_size because calculate with MultiGPU
                gradient_difference_loss_G = tf.reduce_sum(gradient_difference_loss_G) * (1.0 / batch_size)
                gradient_difference_loss_F = tf.reduce_sum(gradient_difference_loss_F) * (1.0 / batch_size)

                # ssim loss
                ssim_loss_G = self.ssim_loss_fn(fake_mr, mr)
                ssim_loss_F = self.ssim_loss_fn(fake_ct, ct)

                # Divide By batch_size because calculate with MultiGPU
                ssim_loss_G = tf.reduce_sum(ssim_loss_G) * (1.0 / batch_size)
                ssim_loss_F = tf.reduce_sum(ssim_loss_F) * (1.0 / batch_size)

                # ssim loss
                perceptual_loss_G = self.perceptual_loss_fn(fake_mr, mr)
                perceptual_loss_F = self.perceptual_loss_fn(fake_ct, ct)

                # VGG model feature loss
                # perceptual_loss_G = tf.reduce_sum(perceptual_loss_G) * (1.0 / batch_size)
                # perceptual_loss_F = tf.reduce_sum(perceptual_loss_F) * (1.0 / batch_size)

                # Total generator loss
                total_loss_G = cycle_loss_G + gradient_difference_loss_G + perceptual_loss_G + ssim_loss_G + gen_G_loss
                total_loss_F = cycle_loss_F + gradient_difference_loss_F + ssim_loss_F + perceptual_loss_F + gen_F_loss

                # Discriminator loss
                disc_X_loss = self.discriminator_loss_fn(disc_real_x, disc_fake_x)
                disc_Y_loss = self.discriminator_loss_fn(disc_real_y, disc_fake_y)

                disc_X_loss = tf.reduce_sum(disc_X_loss) * (1.0 / batch_size)
                disc_Y_loss = tf.reduce_sum(disc_Y_loss) * (1.0 / batch_size)

            # Get the gradients for the generators
            grads_G = tape.gradient(total_loss_G, self.gen_G.trainable_variables)
            grads_F = tape.gradient(total_loss_F, self.gen_F.trainable_variables)

            # Get the gradients for the discriminators
            disc_X_grads = tape.gradient(disc_X_loss, self.disc_X.trainable_variables)
            disc_Y_grads = tape.gradient(disc_Y_loss, self.disc_Y.trainable_variables)

            # Update the weights of the generators
            self.gen_G_optimizer.apply_gradients(
                zip(grads_G, self.gen_G.trainable_variables)
            )
            self.gen_F_optimizer.apply_gradients(
                zip(grads_F, self.gen_F.trainable_variables)
            )

            # Update the weights of the discriminators
            self.disc_X_optimizer.apply_gradients(
                zip(disc_X_grads, self.disc_X.trainable_variables)
            )
            self.disc_Y_optimizer.apply_gradients(
                zip(disc_Y_grads, self.disc_Y.trainable_variables)
            )

            return {
                "G_loss": total_loss_G,
                "F_loss": total_loss_F,
                "D_X_loss": disc_X_loss,
                "D_Y_loss": disc_Y_loss,
            }

        history = strategy.run(
            calculate, args=(ct, mr, self.batch_size,))  # call multi gpu calculate
        total_loss_G = strategy.reduce(tf.distribute.ReduceOp.SUM, history['G_loss'], axis=None)
        total_loss_F = strategy.reduce(tf.distribute.ReduceOp.SUM, history['F_loss'], axis=None)
        disc_X_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, history['D_X_loss'], axis=None)
        disc_Y_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, history['D_Y_loss'], axis=None)

        return {
            "G_loss": total_loss_G,
            "F_loss": total_loss_F,
            "D_X_loss": disc_X_loss,
            "D_Y_loss": disc_Y_loss,
        }
