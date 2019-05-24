import os
from shutil import copy
import numpy as np
from scipy.ndimage import imread
from scipy.misc import imresize
import random


def get_image_batch(filenames, image_dir="./img_align_celeba/", batch_size=32):
    # batch_filenames = random.sample(filenames, batch_size)
    images = []
    num_files = 10000
    for i in range(batch_size):
        images.append(imresize(imread(image_dir + filenames[random.randint(0, num_files)]), (64, 64, 3)))
    return np.array(images)


def get_image_names(image_dir="./img_align_celeba"):
    filenames = []
    for file in os.listdir(image_dir):
        filenames.append(file)
    return filenames


def split_folders(image_dir, new_image_dir, files_per_subdir=1000):
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



split_folders("D:/img_align_celeba/", "D:/img_align_celeba_subdirs/", 1000)




