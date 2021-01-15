import os

from Dicom.Get_Contour_Brain_CT.model.gan import GAN
from Dicom.Get_Contour_Brain_CT.model.generator import Generator
from Dicom.Get_Contour_Brain_CT.model.discriminator import Discriminator
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from random import randint
import tensorflow as tf


class LearningSequence:

    def __init__(self, input_shape):
        if not os.path.exists('./data'):
            os.makedirs('./data')
        self.W = input_shape[0]
        self.H = input_shape[1]
        self.C = input_shape[2]
        self.EPOCHS = 50001
        self.BATCH = 32
        self.CHECKPOINT = 500
        self.model_type = -1

        self.LATENT_SPACE_SIZE = 100
        self.load_MNIST()

        self.generator = Generator(name="Generator", input_shape=input_shape)
        self.discriminator = Discriminator(name="Discriminator", input_shape=input_shape)
        self.gan = GAN(generator=self.generator.model, discriminator=self.discriminator.model)

    def load_MNIST(self, model_type=3):
        allowed_types = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        if self.model_type not in allowed_types:
            print('ERROR: Only Integer Values from -1 to 9 are allowed')

        (self.X_train, self.Y_train), (_, _) = mnist.load_data()
        if self.model_type != -1:
            self.X_train = self.X_train[np.where(self.Y_train == int(self.model_type))[0]]

        # Rescale -1 to 1
        # Find Normalize Function from CV Class
        self.X_train = (np.float32(self.X_train) - 127.5) / 127.5
        self.X_train = np.expand_dims(self.X_train, axis=3)
        return

    def train(self):
        for e in range(self.EPOCHS):
            # Train Discriminator
            # Make the training batch for this model be half real, half noise
            # Grab Real Images for this training batch
            count_real_images = int(self.BATCH / 2)
            starting_index = randint(0, (len(self.X_train) - count_real_images))
            real_images_raw = self.X_train[starting_index: (starting_index + count_real_images)]
            x_real_images = real_images_raw.reshape(count_real_images, self.W, self.H, self.C)
            y_real_labels = np.ones([count_real_images, 1])

            # Grab Generated Images for this training batch

            latent_space_samples = self.sample_latent_space(count_real_images)
            x_generated_images = self.generator.model.predict(latent_space_samples)
            y_generated_labels = np.zeros([self.BATCH - count_real_images, 1])

            # Now, train the discriminator with this batch
            print(np.shape(x_real_images), np.shape(y_real_labels))
            self.discriminator.model.trainable = True
            discriminator_loss = self.discriminator.model.train_on_batch(x_real_images, y_real_labels)
            discriminator_loss += self.discriminator.model.train_on_batch(x_generated_images, y_generated_labels)
            self.discriminator.model.trainable = False
            # Generate Noise
            x_latent_space_samples = self.sample_latent_space(self.BATCH)
            y_generated_labels = np.ones([self.BATCH, 1])
            generator_loss = self.gan.gan_model.train_on_batch(x_latent_space_samples, y_generated_labels)

            print('Epoch: ' + str(int(e)) + ', [Discriminator :: Loss: ' + str(
                discriminator_loss) + '], [ Generator :: Loss: ' + str(generator_loss) + ']')

            if e % self.CHECKPOINT == 0:
                self.plot_checkpoint(e)
        return

    def sample_latent_space(self, instances):
        return np.random.normal(0, 1, (instances, 28, 28, 1))

    def plot_checkpoint(self, e):
        filename = "./data/sample_" + str(e) + ".png"

        noise = self.sample_latent_space(16)
        images = self.generator.model.predict(noise)

        plt.figure(figsize=(10, 10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i + 1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.H, self.W])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close('all')
        return


if __name__ == '__main__':
    trainer = LearningSequence((28, 28, 1))
    # trainer.train()
    # print(Learning_Sequence.generator.model.summary())
    # print(Learning_Sequence.generator.model.predict(np.ones((1, 512, 512, 1))))
