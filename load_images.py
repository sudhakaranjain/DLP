import os
import numpy as np
from scipy.ndimage import imread
from scipy.misc import imresize
from keras.preprocessing.image import ImageDataGenerator
import random


def get_image_batch(filenames, image_dir="./img_align_celeba/", batch_size=32):
    batch_filenames = random.sample(filenames, batch_size)
    images = []
    for name in batch_filenames:
        images.append(imresize(imread(image_dir + name), (64, 64, 3)))
    return np.array(images)


def get_image_names(image_dir="./img_align_celeba"):
    filenames = []
    for file in os.listdir(image_dir):
        filenames.append(file)
    return filenames

