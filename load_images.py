import os
from shutil import copy
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import random
from PIL import Image

IMG_SIZE = 64


def get_image_batch(image_dir="./img_align_celeba/", batch_size=32, val=False):
    images = []

    if val:
        image_dir = image_dir + "val/"
    else:
        image_dir = image_dir + "train/"

    num_subdirs = len(os.listdir(image_dir))
    subdir = random.randint(0, num_subdirs - 1)

    filenames = []
    for file in os.listdir(image_dir + str(subdir)):
        filenames.append(file)

    for i in range(batch_size):
        file_index = random.randint(0, len(filenames) - 1)
        image = Image.open(image_dir + str(subdir) + "/" + filenames[file_index])
        image = image.resize((IMG_SIZE, IMG_SIZE))

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
        train_files = files[split_idx:]

        count = 0
        subdir = 0
        os.makedirs(new_image_dir + "train/" + str(subdir))
        for file in train_files:
            copy(image_dir + file, new_image_dir + "train/" + str(subdir) + "/" + file)
            count += 1
            if count >= files_per_subdir:
                subdir += 1
                os.makedirs(new_image_dir + "train/" + str(subdir))
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
def remove_hole_image(image, type):
    if(type=='centre'):
        scale = 0.25
        DIM = IMG_SIZE * scale
        start_x, start_y = int(IMG_SIZE - DIM)
        for i in range(start_y, start_y + DIM + 1):
            image[i][start_x:start_x + DIM] = 255
    elif(type=='rect'):
        scale = 0.25
        hole_height = random.randint(int((IMG_SIZE-30)*scale), int((IMG_SIZE+10)*scale))
        hole_width = random.randint(int((IMG_SIZE-30)*scale), int((IMG_SIZE+10)*scale))
        start_y = random.randint(1, IMG_SIZE - hole_height) - 1
        offset_x_axis = random.randint(1, IMG_SIZE - hole_width) - 1
        for i in range(start_y, start_y + hole_height + 1):
            image[i][offset_x_axis:offset_x_axis + hole_width] = 255
    elif(type=='random'):
        perc_blocked = 0.2
        for i in range(IMG_SIZE + 1):
            for j in range(IMG_SIZE + 1):
                if(np.random.random() < perc_blocked):
                    image[i][j] = 255
    elif(type=='left'):
        for i in range(0, IMG_SIZE / 2):
            for i in range(0, IMG_SIZE):
                image[i][j] = 255
    elif(type == 'right'):
        for j in range(IMG_SIZE / 2, IMG_SIZE):
            for i in range(0, IMG_SIZE):
                image[i][j] = 255
    elif(type == 'top'):
        for i in range(0, IMG_SIZE / 2):
            for j in range(0, IMG_SIZE):
                image[i][j] = 255
    elif(type == 'bottom'):
        for i in range(IMG_SIZE / 2, IMG_SIZE):
            for j in range(0, IMG_SIZE):
                image[i][j] = 255
    else:
        print("No valid hole settings detected")
    return image
