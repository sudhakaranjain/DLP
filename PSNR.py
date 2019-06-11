import numpy as np
import math
import os
import unet
from unet import swish
from unet import Swish
from keras.models import Model, load_model
from keras.utils.generic_utils import get_custom_objects
from load_images import remove_hole_image
from PIL import Image

#dataset subdirs location
dataset = "./dataset/celeba-dataset/img_align_celeba_subdirs"
#hole type
hole_type = 'centre'
#mages
images = []
generated_images = []

IMG_SIZE = 64

# Help function to calculate PSNR
def psnr(origin, test):

    mse = np.mean((origin - test) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# Calculate mean Peak Signal-to-Noise Ratio
if __name__ == '__main__':
    get_custom_objects().update({'swish': Swish(swish)})
    # Pick random validation set
    num_subdirs = len(os.listdir(dataset+"/val"))
    valdir = dataset+"/val/"+str(np.random.randint(0, num_subdirs-1))
    for file in os.listdir(valdir):
        image = Image.open(valdir+"/"+file)
        image = image.resize((IMG_SIZE, IMG_SIZE))
        images.append(np.array(image))
    holed_images = images.copy()
    for i in range(0, len(images)):
        holed_images[i] = remove_hole_image(images[i], hole_type)
    try:
        model = load_model("unet.h5")
        print("UNet model found, starting calculations")
        generated_images = model.predict(np.array(holed_images))
        psnr_total = 0
        for i in range(0, len(images)):
            psnr_total += psnr(images[i], generated_images[i])
        result = "PSNR UNet: " + str(psnr_total / len(images))
        print(result)
    except OSError:
        print("No UNet model found!")