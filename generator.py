import os

import numpy as np
from keras.layers import BatchNormalization, Activation
from keras.layers import Dense, Reshape, Flatten, Dropout, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential
from keras.optimizers import Adam
from scipy.misc import imsave

from load_images import get_image_batch


class GAN:
    def __init__(self):
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.noise_dim = 100

        optimizer = Adam(lr=0.0002, beta_1=0.5)

        self.discriminator = self.discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        self.generator = self.generator()

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        self.combined = Sequential()
        self.combined.add(self.generator)
        self.combined.add(self.discriminator)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def generator(self):
        model = Sequential()

        model.add(Dense(4 * 4 * 1024, activation="relu", input_dim=self.noise_dim))
        model.add(Reshape((4, 4, 1024)))
        model.add(UpSampling2D())
        model.add(Conv2D(512, kernel_size=3, padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(256, kernel_size=3, padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()
        return model

    def discriminator(self):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())

        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        return model

    def train(self, epochs, image_dir, batch_size=128, sample_interval=50):
        # Adversarial ground truths
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            images = get_image_batch(image_dir, batch_size)
            images = images / 127.5 - 1.

            noise = np.random.normal(0, 1, (batch_size, self.noise_dim))

            # Generate a batch of new images
            gen_images = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(images, real)
            d_loss_fake = self.discriminator.train_on_batch(gen_images, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, (batch_size, self.noise_dim))

            # target: output from noise vector is alwasy classified as real by discriminator
            # this trains the generator only, as the discriminator is not trainable
            g_loss = self.combined.train_on_batch(noise, real)

            print("Epoch:", epoch, "D_loss_r:", d_loss_real[0], "D_loss_f:", d_loss_fake[0], "G_loss:", g_loss)
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch, num_images=10):
        noise = np.random.normal(0, 1, (num_images, self.noise_dim))
        gen_images = self.generator.predict(noise)

        os.makedirs("./images/" + str(epoch) + "/", exist_ok=True)

        # Rescale images
        gen_images = 0.5 * gen_images + 0.5

        for x in range(num_images):
            imsave("./images/" + str(epoch) + "/" + str(x) + ".png", gen_images[x, :, :, :])


if __name__ == '__main__':
    batch_size = 64
    gan = GAN()
    image_dir = "D:/img_align_celeba/"
    gan.train(epochs=1000, image_dir=image_dir, batch_size=batch_size, sample_interval=100)
