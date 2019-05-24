import os
from shutil import copy
import numpy as np
from scipy.ndimage import imread
from scipy.misc import imresize
from keras.preprocessing.image import ImageDataGenerator
import random


def get_image_batch(image_dir="./img_align_celeba/", batch_size=32, split_dirs=True):
    images = []

    if not split_dirs:
        filenames = []
        for file in os.listdir(image_dir):
            filenames.append(file)
        batch_filenames = random.sample(filenames, batch_size)
        for name in batch_filenames:
            images.append(imresize(imread(image_dir + name), (64, 64, 3)))
    else:
        # get dirs
        dirs = []
        num_subdirs = 0
        for _ in os.listdir():
            num_subdirs += 1
        subdir = random.randint(0, num_subdirs - 1)

        filenames = []
        for file in os.listdir(image_dir + str(subdir)):
            filenames.append(file)

        for i in range(batch_size):
            file_index = random.randint(0, len(filenames) - 1)
            images.append(imresize(imread(image_dir + str(subdir) + "/" + filenames[file_index]), (64, 64, 3)))

    return np.array(images)


def get_image_names(image_dir="./img_align_celeba"):
    filenames = []
    for file in os.listdir(image_dir):
        filenames.append(file)
    return filenames


def split_folders(image_dir, new_image_dir, files_per_subdir=1000):
    if not os.path.isdir(new_image_dir):
        count = 0
        subdir = 0
        os.makedirs(new_image_dir + str(subdir))
        for file in os.listdir(image_dir):
            copy(image_dir + file, new_image_dir + str(subdir) + "/" + file)
            count += 1
            if count >= files_per_subdir:
                subdir += 1
                os.makedirs(new_image_dir + str(subdir))
                count = 0


# Takes an image, returns an image with a white hole in it. Parameters can be set in the function call.
def remove_hole_image(image, hole_heigth=20, hole_width=20, starting_row=22, offset_x_axis=22):
    for i in range(starting_row, starting_row + hole_heigth + 1):
        image[i][offset_x_axis:offset_x_axis + hole_width] = 255
    return image
