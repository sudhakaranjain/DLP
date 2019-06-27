import random, os

import numpy as np
import matplotlib.pyplot as plt
import argparse
from keras.layers import BatchNormalization, Activation
from keras.layers import Dense, Flatten, Dropout, ZeroPadding2D, Input, MaxPooling2D, concatenate
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
# For adding new activation function
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

from load_images import get_image_batch, split_folders, remove_hole_image
import time
import datetime
import pandas as pd

start_time_for_stamp = datetime.datetime.now()
start_time_timestamp = start_time_for_stamp.strftime("%Y-%m-%d %H%M")
start_time = time.time()


class Swish(Activation):

    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'


# Swish activtion function
def swish(x):
    return x * K.sigmoid(x)


class GAN:
    def __init__(self, image_dir, activation_function='swish'):
        self.activation_function = activation_function

        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.noise_dim = 100
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('-t', '--type', type=str,
                                 choices=(['centre',
                                           'rect',
                                           'random',
                                           'left',
                                           'right',
                                           'top',
                                           'bottom',
                                           ]),
                                 default='centre')
        self.args = self.parser.parse_args()
        self._image_dir = image_dir
        # to prevent having to load the image filenames every epoch, the list of filenames is retrieved once and then stored

        optimizer = Adam(lr=0.0002, beta_1=0.5)

        self.discriminator = self.create_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        self.generator = self.create_generator()

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        self.combined = Sequential()
        self.combined.add(self.generator)
        self.combined.add(self.discriminator)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

        self.train_loss_history_discriminator = []
        self.train_loss_history_generator = []

    def create_generator(self):
        try:
            model = load_model("generator.h5")
        except OSError:
            input_img = Input(shape=self.img_shape)  # adapt this if using `channels_first` image data format

            conv1 = Conv2D(16, (3, 3), activation=self.activation_function, padding='same')(input_img)
            pool1 = MaxPooling2D((2, 2), padding='same')(conv1)
            conv2 = Conv2D(32, (3, 3), activation=self.activation_function, padding='same')(pool1)
            pool2 = MaxPooling2D((2, 2), padding='same')(conv2)
            conv3 = Conv2D(64, (3, 3), activation=self.activation_function, padding='same')(pool2)
            pool3 = MaxPooling2D((2, 2), padding='same')(conv3)

            conv4 = Conv2D(64, (3, 3), activation=self.activation_function, padding='same')(pool3)
            up1 = UpSampling2D((2, 2))(conv4)
            merge1 = concatenate([conv3, up1])
            conv5 = Conv2D(32, (3, 3), activation=self.activation_function, padding='same')(merge1)
            up2 = UpSampling2D((2, 2))(conv5)
            merge2 = concatenate([conv2, up2])
            conv6 = Conv2D(16, (3, 3), activation=self.activation_function, padding='same')(merge2)
            up3 = UpSampling2D((2, 2))(conv6)
            merge3 = concatenate([conv1, up3])
            conv7 = Conv2D(3, (3, 3), activation='tanh', padding='same')(merge3)

            model = Model(input_img, conv7)
            model.compile(optimizer='adam', loss='mse')
        return model

    def create_discriminator(self):
        try:
            model = load_model("discriminator.h5")
        except OSError:
            model = Sequential()

            model.add(
                Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same",
                       activation=self.activation_function))
            model.add(Dropout(0.25))
            model.add(Conv2D(64, kernel_size=3, strides=2, padding="same", activation=self.activation_function))
            model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Dropout(0.25))
            model.add(Conv2D(128, kernel_size=3, strides=2, padding="same", activation=self.activation_function))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Dropout(0.25))
            model.add(Conv2D(256, kernel_size=3, strides=1, padding="same", activation=self.activation_function))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Dropout(0.25))
            model.add(Flatten())

            model.add(Dense(1, activation='sigmoid'))
            model.summary()
        return model

    def train(self, epochs, batch_size=128, sample_interval=50, train_until_no_improvement=False,
              improvement_threshold=0.001):
        # Adversarial ground truths
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            images = get_image_batch(self._image_dir, batch_size)  # Get train ims
            images_holes = images + 0
            for index in range(len(images)):
                images_holes[index, :, :, :] = remove_hole_image(images_holes[index, :, :, :], type=self.args.type)
            images = images / 127.5 - 1.
            images_holes = images_holes / 127.5 - 1.

            self.generator.train_on_batch(images_holes, images)

            # Generate a batch of new images
            gen_images = self.generator.predict(images_holes)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(images, real)
            d_loss_fake = self.discriminator.train_on_batch(gen_images, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # target: output from noise vector is alwasy classified as real by discriminator
            # this trains the generator only, as the discriminator is not trainable
            g_loss = self.combined.train_on_batch(images_holes, real)

            print("Epoch:", epoch, "D_loss_r:", d_loss_real[0], "D_loss_f:", d_loss_fake[0], "G_loss:", g_loss)

            if epoch % sample_interval == 0:
                images = get_image_batch(self._image_dir, batch_size, val=True)  # Get val ims
                images_holes = images + 0
                for index in range(len(images)):
                    images_holes[index, :, :, :] = remove_hole_image(images_holes[index, :, :, :], type=self.args.type)
                images = images / 127.5 - 1.
                images_holes = images_holes / 127.5 - 1.
                decoded_imgs = self.generator.predict(images_holes)

                remaining_time_estimate = (((time.time() - start_time) / 60) / (epoch + 1)) * (
                        (epochs + 1) - (epoch + 1))
                print("Estimated time remaining: {:.4} min".format(
                    remaining_time_estimate) + "| Time elapsed: {:.4} min".format(((time.time() - start_time) / 60)))
                os.makedirs(output_dir + "/images/", exist_ok=True)

                self.train_loss_history_discriminator.append(np.mean(d_loss))
                self.train_loss_history_generator.append(g_loss)

                n = 10
                plt.figure(figsize=(20, 4))
                for i in range(n):
                    # display original
                    image_idx = random.randint(0, len(decoded_imgs) - 1)
                    ax = plt.subplot(2, n, i + 1)
                    plt.imshow(((images_holes[image_idx].reshape(64, 64, 3) + 1) * 127.5).astype(np.uint8))
                    plt.gray()
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)

                    # display reconstruction
                    ax = plt.subplot(2, n, i + n + 1)
                    plt.imshow(((decoded_imgs[image_idx].reshape(64, 64, 3) + 1) * 127.5).astype(np.uint8))
                    plt.gray()
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                plt.savefig(output_dir + "/images/" + str(epoch) + ".png")
                plt.close()

                self.generator.save(output_dir + "generator.h5")
                self.discriminator.save(output_dir + "discriminator.h5")

                # Only considers the generator
                if train_until_no_improvement:
                    if len(self.train_loss_history_generator) <= sample_interval:  # First run through loop
                        last_mean_loss = 9999
                        current_mean_loss = 999
                    else:
                        last_mean_loss = current_mean_loss
                        current_mean_loss = np.mean(
                            self.train_loss_history_generator[-sample_interval])  # Take last x items from the list
                    if (last_mean_loss - current_mean_loss) < improvement_threshold:
                        in_a_row += 1
                        print("No improvement in a row: " + str(in_a_row))
                        if in_a_row >= 10:
                            return  # Break out of the function
                    else:
                        in_a_row = 0


def visualize_results(self, sample_interval):
    plt.figure()
    epochs = len(self.train_loss_history_discriminator)
    plt.plot(range(1, epochs + 1), self.train_loss_history_discriminator, label="Discriminator loss")
    plt.plot(range(1, epochs + 1), self.train_loss_history_generator, label="Generator loss")
    plt.legend()
    plt.xlabel("Epoch x" + str(sample_interval))
    plt.ylabel("Error")
    plt.title("Error over time")
    plt.savefig(output_dir + 'plot.png')


def save_loss_data(self):
    d = {'Discriminator loss': self.train_loss_history_discriminator,
         'Generator loss': self.train_loss_history_generator}
    dataframe = pd.DataFrame(d)
    dataframe.to_csv(output_dir + "loss_with_" + self.activation_function + ".csv", index=True, index_label="Epoch")


if __name__ == '__main__':
    # Get swish to work
    get_custom_objects().update({'swish': Swish(swish)})
    # To make reading the files faster, they need to be divided into subdirectories.
    # split_folders("./img_align_celeba/", "./img_align_celeba_subdirs/", 1000)
    # batch_size = 128
    batch_size = 256
    # image_dir = "./img_align_celeba_subdirs/"
    image_dir = "./celeba-dataset/img_align_celeba_subdirs/"
    output_dir = "./GAN/" + start_time_timestamp + "/"
    activation_function = 'relu'  # either swish or relu (case sensitive)
    sample_interval = 5

    gan = GAN(image_dir, activation_function)
    gan.train(epochs=25, batch_size=batch_size, sample_interval=sample_interval, train_until_no_improvement=True,
              improvement_threshold=0.01)
    visualize_results(gan, sample_interval)
    save_loss_data(gan)
