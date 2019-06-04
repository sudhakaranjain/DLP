import random, os

import numpy as np
import matplotlib.pyplot as plt
from keras.layers import BatchNormalization, Activation
from keras.layers import Dense, Reshape, Flatten, Dropout, ZeroPadding2D, Input, Conv2D, MaxPooling2D, UpSampling2D, \
    concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam

from load_images import get_image_batch, split_folders, remove_hole_image
import time

start_time = time.time()


class GAN:
    def __init__(self, image_dir):
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.noise_dim = 100

        self._image_dir = image_dir
        # to prevent having to load the image filenames every epoch, the list of filenames is retrieved once and then stored

        optimizer = Adam(lr=0.0002, beta_1=0.5)

        input_img_d, output_d = self.create_discriminator()
        self.discriminator = Model(input_img_d, output_d)
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        input_img_g, output_g = self.create_generator()
        self.generator = Model(input_img_g, output_g)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        self.combined = Model(inputs=[input_img_g, input_img_d], outputs=[output_g, output_d])
        self.combined.compile(loss=['mae', 'binary_crossentropy'], optimizer=optimizer)

    def create_generator(self):
        try:
            model = load_model("generator.h5")
        except OSError:
            input_img = Input(shape=self.img_shape)  # adapt this if using `channels_first` image data format

            conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
            pool1 = MaxPooling2D((2, 2), padding='same')(conv1)
            conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
            pool2 = MaxPooling2D((2, 2), padding='same')(conv2)
            conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
            pool3 = MaxPooling2D((2, 2), padding='same')(conv3)

            conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool3)
            up1 = UpSampling2D((2, 2))(conv4)
            merge1 = concatenate([conv3, up1])
            conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(merge1)
            up2 = UpSampling2D((2, 2))(conv5)
            merge2 = concatenate([conv2, up2])
            conv6 = Conv2D(16, (3, 3), activation='relu', padding='same')(merge2)
            up3 = UpSampling2D((2, 2))(conv6)
            merge3 = concatenate([conv1, up3])
            conv7 = Conv2D(3, (3, 3), activation='tanh', padding='same')(merge3)

        return input_img, conv7

    def create_discriminator(self):
        try:
            model = load_model("discriminator.h5")
        except OSError:
            model = Sequential()
            input_img = Input(self.img_shape)
            x = Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same")(input_img)
            x = LeakyReLU(alpha=0.2)(x)
            x = Dropout(0.25)(x)
            x = Conv2D(64, kernel_size=3, strides=2, padding="same")(x)
            x = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = Dropout(0.25)(x)
            x = Conv2D(128, kernel_size=3, strides=2, padding="same")(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = Dropout(0.25)(x)
            x = Conv2D(256, kernel_size=3, strides=1, padding="same")(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = Dropout(0.25)(x)
            x = Flatten()(x)

            x = Dense(1, activation='sigmoid')(x)
        return input_img, x

    def train(self, epochs, batch_size=128, sample_interval=50):
        # Adversarial ground truths
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            images = get_image_batch(self._image_dir, batch_size)  # Get train ims
            images_holes = images + 0
            for index in range(len(images)):
                images_holes[index, :, :, :] = remove_hole_image(images_holes[index, :, :, :])
            images = images / 127.5 - 1.
            images_holes = images_holes / 127.5 - 1.

            # Generate a batch of new images
            gen_images = self.generator.predict(images_holes)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(images, real)
            d_loss_fake = self.discriminator.train_on_batch(gen_images, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # target: output from noise vector is alwasy classified as real by discriminator
            # this trains the generator only, as the discriminator is not trainable
            g_loss = self.combined.train_on_batch([images_holes, images_holes], [images, real])

            print("Epoch:", epoch, "D_loss_r:", d_loss_real[0], "D_loss_f:", d_loss_fake[0], "G_loss:", g_loss)

            if epoch % sample_interval == 0:
                images = get_image_batch(self._image_dir, batch_size, val=True)  # Get val ims
                images_holes = images + 0
                for index in range(len(images)):
                    images_holes[index, :, :, :] = remove_hole_image(images_holes[index, :, :, :])
                images = images / 127.5 - 1.
                images_holes = images_holes / 127.5 - 1.
                decoded_imgs = self.generator.predict(images_holes)

                remaining_time_estimate = (((time.time() - start_time) / 60) / (epoch + 1)) * (
                            (epochs + 1) - (epoch + 1))
                print("Estimated time remaining: {:.4} min".format(
                    remaining_time_estimate) + "| Time elapsed: {:.4} min".format(((time.time() - start_time) / 60)))
                os.makedirs("./images_gen/", exist_ok=True)

                n = 10
                plt.figure(figsize=(20, 4))
                for i in range(n):
                    # display original
                    image_idx = random.randint(0, len(decoded_imgs) - 1)
                    ax = plt.subplot(2, n, i + 1)
                    plt.imshow(((images[image_idx].reshape(64, 64, 3) + 1) * 127.5).astype(np.uint8))
                    plt.gray()
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)

                    # display reconstruction
                    ax = plt.subplot(2, n, i + n + 1)
                    plt.imshow(((decoded_imgs[image_idx].reshape(64, 64, 3) + 1) * 127.5).astype(np.uint8))
                    plt.gray()
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                plt.savefig("./images_gen/" + str(epoch) + ".png")

        self.generator.save("generator.h5")
        self.discriminator.save("discriminator.h5")


if __name__ == '__main__':
    # To make reading the files faster, they need to be divided into subdirectories.
    # split_folders("./img_align_celeba/", "./img_align_celeba_subdirs/", 1000)
    batch_size = 128
    image_dir = "D:/img_align_celeba_subdirs/"
    gan = GAN(image_dir)
    gan.train(epochs=10000, batch_size=batch_size, sample_interval=25)
