import os
from shutil import copy
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import random
from PIL import Image


def get_image_batch(image_dir="./img_align_celeba/", batch_size=32, val=False):
    images = []

    if val:
        image_dir = image_dir + "val/"
    else:
        image_dir = image_dir + "test/"

    num_subdirs = len(os.listdir(image_dir))
    subdir = random.randint(0, num_subdirs - 1)

    filenames = []
    for file in os.listdir(image_dir + str(subdir)):
        filenames.append(file)

    for i in range(batch_size):
        file_index = random.randint(0, len(filenames) - 1)
        image = Image.open(image_dir + str(subdir) + "/" + filenames[file_index])
        image = image.resize((64, 64))

        images.append(np.array(image))

    return np.array(images)


def get_image_names(image_dir="./img_align_celeba"):
    filenames = []
    for file in os.listdir(image_dir):
        filenames.append(file)
    return filenames


def split_folders(image_dir, new_image_dir, files_per_subdir=1000, val_split=0.2):
    if not os.path.isdir(new_image_dir):
        files = os.listdir(image_dir)
        num_files = len(files)
        split_idx = int(val_split * num_files)
        val_files = files[0:split_idx]
        test_files = files[split_idx:]

        count = 0
        subdir = 0
        os.makedirs(new_image_dir + "test/" + str(subdir))
        for file in test_files:
            copy(image_dir + file, new_image_dir + "test/" + str(subdir) + "/" + file)
            count += 1
            if count >= files_per_subdir:
                subdir += 1
                os.makedirs(new_image_dir + "test/" + str(subdir))
                count = 0

        count = 0
        subdir = 0
        os.makedirs(new_image_dir + "val/" + str(subdir))
        for file in val_files:
            copy(image_dir + file, new_image_dir + "val/" + str(subdir) + "/" + file)
            count += 1
            if count >= files_per_subdir:
                subdir += 1
                os.makedirs(new_image_dir + "val/" + str(subdir))
                count = 0

# Takes an image, returns an image with a white hole in it. Parameters can be set in the function call.
def remove_hole_image(image, hole_heigth=20, hole_width=20, starting_row=22, offset_x_axis=22):
    for i in range(starting_row, starting_row + hole_heigth + 1):
        image[i][offset_x_axis:offset_x_axis + hole_width] = 255
    return image
