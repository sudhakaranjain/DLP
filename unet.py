import random, os

import matplotlib.pyplot as plt
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.models import Model, load_model

from load_images import get_image_batch, split_folders, remove_hole_image


class Unet:
    def __init__(self, image_dir):
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self._image_dir = image_dir

        self.model = self.create_model()

    def create_model(self):
        try:
            model = load_model("unet.h5")
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
            conv7 = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(merge3)

            model = Model(input_img, conv7)
            model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, epochs, batch_size=128, sample_interval=50):
        for x in range(epochs):
            images = get_image_batch(self._image_dir, batch_size)  # Get train ims
            images_holes = images + 0
            for index in range(len(images)):
                images_holes[index, :, :, :] = remove_hole_image(images_holes[index, :, :, :])
            images = images / 127.5 - 1.
            images_holes = images_holes / 127.5 - 1.

            self.model.fit(images_holes, images, verbose=2)

            if x % sample_interval == 0:
                images = get_image_batch(self._image_dir, batch_size, val=True)  # Get val ims
                images_holes = images + 0
                for index in range(len(images)):
                    images_holes[index, :, :, :] = remove_hole_image(images_holes[index, :, :, :])
                images = images / 127.5 - 1.
                images_holes = images_holes / 127.5 - 1.
                decoded_imgs = self.model.predict(images_holes)
                os.makedirs("./images_unet/", exist_ok=True)

                n = 10
                plt.figure(figsize=(20, 4))
                for i in range(n):
                    # display original
                    image_idx = random.randint(0, len(decoded_imgs))
                    ax = plt.subplot(2, n, i + 1)
                    plt.imshow(images[image_idx].reshape(64, 64, 3))
                    plt.gray()
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)

                    # display reconstruction
                    ax = plt.subplot(2, n, i + n + 1)
                    plt.imshow(decoded_imgs[image_idx].reshape(64, 64, 3))
                    plt.gray()
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                plt.savefig("./images_unet/" + str(x) + ".png")

        self.model.save("unet.h5")


if __name__ == '__main__':
    # To make reading the files faster, they need to be divided into subdirectories.
    split_folders("D:/img_align_celeba/", "D:/img_align_celeba_subdirs/", 1000)
    batch_size = 4096
    image_dir = "D:/img_align_celeba_subdirs/"
    model = Unet(image_dir)
    model.train(epochs=10, batch_size=batch_size, sample_interval=5)
