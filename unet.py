import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import backend as K
from keras.callbacks import History
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Activation
from keras.models import Model, load_model
from keras.utils.generic_utils import get_custom_objects
from keras_contrib.losses import DSSIMObjective

from load_images import get_image_batch, split_folders, remove_hole_image


class Swish(Activation):

    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'


def swish(x):
    return K.sigmoid(x) * x


def custom_loss(y_pred, y_true):
    # Doesn't work yet
    dssim = DSSIMObjective()
    return dssim(y_true, y_pred)


class Unet:
    def __init__(self, image_dir, activation='relu'):
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.activation = activation
        self._image_dir = image_dir

        self.model = self.create_model()
        self.history = History()
        self.train_loss_history = []

    def create_model(self):
        try:
            custom_objects = {'DSSIMObjective': DSSIMObjective()}
            model = load_model("unet.h5", custom_objects)
        except OSError:
            input_img = Input(shape=self.img_shape)  # adapt this if using `channels_first` image data format

            convd1 = Conv2D(16, (3, 3), activation=self.activation, padding='same')(input_img)
            down1 = MaxPooling2D((2, 2), padding='same')(convd1)
            convd2 = Conv2D(32, (3, 3), activation=self.activation, padding='same')(down1)
            down2 = MaxPooling2D((2, 2), padding='same')(convd2)
            convd3 = Conv2D(64, (3, 3), activation=self.activation, padding='same')(down2)
            down3 = MaxPooling2D((2, 2), padding='same')(convd3)
            convd4 = Conv2D(128, (3, 3), activation=self.activation, padding='same')(down3)
            down4 = MaxPooling2D((2, 2), padding='same')(convd4)

            convu1 = Conv2D(128, (3, 3), activation=self.activation, padding='same')(down4)
            up1 = UpSampling2D((2, 2))(convu1)
            merge1 = concatenate([convd4, up1])
            convu2 = Conv2D(64, (3, 3), activation=self.activation, padding='same')(merge1)
            up2 = UpSampling2D((2, 2))(convu2)
            merge2 = concatenate([convd3, up2])
            convu3 = Conv2D(32, (3, 3), activation=self.activation, padding='same')(merge2)
            up3 = UpSampling2D((2, 2))(convu3)
            merge3 = concatenate([convd2, up3])
            convu4 = Conv2D(16, (3, 3), activation=self.activation, padding='same')(merge3)
            up4 = UpSampling2D((2, 2))(convu4)
            merge4 = concatenate([convd1, up4])

            convu4 = Conv2D(3, (3, 3), activation='tanh', padding='same')(merge4)
            convu4_b = Conv2D(3, (3, 3), activation='tanh', padding='same')(merge4)

            model = Model(input_img, [convu4, convu4_b])
            model.compile(optimizer='adam', loss=['mse', DSSIMObjective()])
        return model

    def train(self, epochs, batch_size=128, sample_interval=50):

        for x in range(epochs):
            images = get_image_batch(self._image_dir, batch_size)  # Get train ims
            images_holes = images + 0
            for index in range(len(images)):
                images_holes[index, :, :, :] = remove_hole_image(images_holes[index, :, :, :], type='rect')
            images = images / 127.5 - 1.
            images_holes = images_holes / 127.5 - 1.

            self.model.fit(images_holes, [images, images], verbose=2, callbacks=[self.history])
            #TODO implement saving our custom loss data
            self.train_loss_history.append(self.model.history.history['loss'][0]) # Bit hacky since we use our own loop to loop through epochs


            if x % sample_interval == 0:
                images = get_image_batch(self._image_dir, batch_size, val=True)  # Get val ims
                images_holes = images + 0
                for index in range(len(images)):
                    images_holes[index, :, :, :] = remove_hole_image(images_holes[index, :, :, :], type='rect')
                images = images / 127.5 - 1.
                images_holes = images_holes / 127.5 - 1.
                decoded_imgs = self.model.predict(images_holes)[0]
                os.makedirs("./images_unet/", exist_ok=True)



                n = 10
                plt.figure(figsize=(20, 6))
                for i in range(n):
                    # image with hole
                    image_idx = random.randint(0, len(decoded_imgs))
                    ax = plt.subplot(3, n, i + 1)
                    plt.imshow(((images_holes[image_idx].reshape(128, 128, 3) + 1) * 127.5).astype(np.uint8))
                    plt.gray()
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)

                    # original            print(image.shape)
                    #             print(np.mean(image))
                    ax = plt.subplot(3, n, i + n + 1)
                    plt.imshow(((images[image_idx].reshape(128, 128, 3) + 1) * 127.5).astype(np.uint8))
                    plt.gray()
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)

                    # reconstruction
                    ax = plt.subplot(3, n, i + n + n + 1)
                    plt.imshow(((decoded_imgs[image_idx].reshape(128, 128, 3) + 1) * 127.5).astype(np.uint8))
                    plt.gray()
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                plt.savefig("./images_unet/" + str(x) + ".png")
                self.model.save("unet.h5")


def visualize_results(model):
    plt.figure()
    epochs = len(model.train_loss_history)
    plt.plot(range(1, epochs+1), model.train_loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Mean square error")
    plt.title("MSE over time")
    plt.xticks(np.arange(1, epochs + 1, 1.0)) # Only use integers for x-axis values
    plt.savefig('./images_unet/plot.png')

def save_loss_data(model):
    dataframe = pd.DataFrame(model.train_loss_history, columns=['MSE loss during training'])
    dataframe.to_csv("./images_unet/loss_with_" + model.activation + ".csv", index=True, index_label = "Epoch")

if __name__ == '__main__':
    get_custom_objects().update({'swish': Swish(swish)})
    # To make reading the files faster, they need to be divided into subdirectories.
    # split_folders("./celeba-dataset/img_align_celeba/", "./celeba-dataset/img_align_celeba_subdirs/", 1000)
    batch_size = 4096
    #image_dir = "D:/img_align_celeba_subdirs/"
    image_dir = "./celeba-dataset/img_align_celeba_subdirs/"
    #output_dir = "./images_unet/"
    model = Unet(image_dir, 'swish')
    model.train(1, batch_size=batch_size, sample_interval=5)
    visualize_results(model)
    save_loss_data(model)
