import keras
import tensorflow as tf
import numpy as np
from loss import Loss
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Reshape


class CycleGan(keras.Model):
    def __init__(
            self,
            generator_G,
            generator_F,
            discriminator_X,
            discriminator_Y,
            lambda_cycle=10.0
    ):
        super(CycleGan, self).__init__()
        self.gen_G = generator_G
        self.gen_F = generator_F
        self.disc_X = discriminator_X
        self.disc_Y = discriminator_Y
        self.lambda_cycle = lambda_cycle
        self.loss_function = Loss()
        self.number = 0

    def compile(
            self,
            gen_G_optimizer,
            gen_F_optimizer,
            disc_X_optimizer,
            disc_Y_optimizer
    ):
        super(CycleGan, self).compile()
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
    def train_step(self, ct, mr):
        with tf.GradientTape(persistent=True) as tape:
            # Horse to fake zebra
            fake_mr = self.gen_G(ct, training=True)
            # Zebra to fake horse -> y2x
            fake_ct = self.gen_F(mr, training=True)
            # Cycle (Horse to fake zebra to fake horse): x -> y -> x
            cycled_x = self.gen_F(fake_mr, training=True)
            # Cycle (Zebra to fake horse to fake zebra) y -> x -> y
            cycled_y = self.gen_G(fake_ct, training=True)

            # Discriminator output
            disc_real_x = self.disc_X(ct, training=True)
            disc_fake_x = self.disc_X(fake_ct, training=True)

            disc_real_y = self.disc_Y(mr, training=True)
            disc_fake_y = self.disc_Y(fake_mr, training=True)

            # Generator adverserial loss
            gen_G_loss = self.generator_loss_fn(disc_fake_y)
            gen_F_loss = self.generator_loss_fn(disc_fake_x)

            # Generator cycle loss
            cycle_loss_G = self.cycle_loss_fn(mr, cycled_y, self.lambda_cycle)
            cycle_loss_F = self.cycle_loss_fn(ct, cycled_x, self.lambda_cycle)

            gradient_difference_loss_G = self.gradient_difference_loss(fake_mr, mr)
            gradient_difference_loss_F = self.gradient_difference_loss(fake_ct, ct)

            ssim_loss_G = self.ssim_loss_fn(fake_mr, mr)
            ssim_loss_F = self.ssim_loss_fn(fake_ct, ct)

            perceptual_loss_G = self.perceptual_loss_fn(fake_mr, mr)
            perceptual_loss_F = self.perceptual_loss_fn(fake_ct, ct)

            # Total generator loss
            total_loss_G = cycle_loss_G + gradient_difference_loss_G + perceptual_loss_G + ssim_loss_G + gen_G_loss
            total_loss_F = cycle_loss_F + gradient_difference_loss_F + ssim_loss_F + perceptual_loss_F + gen_F_loss

            # Discriminator loss
            disc_X_loss = self.discriminator_loss_fn(disc_real_x, disc_fake_x)
            disc_Y_loss = self.discriminator_loss_fn(disc_real_y, disc_fake_y)

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
