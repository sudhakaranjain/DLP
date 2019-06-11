import numpy
import math
import os
import unet
from keras.models import Model, load_model
from load_images import remove_hole_image
from PIL import Image

#dataset subdirs location
dataset = "./dataset/celeba-dataset/img_align_celeba_subdirs"
#hole type
hole_type = 'centre'
#mages
images = []
generated_images = []

# Help function to calculate PSNR
def psnr(origin, test):

    mse = numpy.mean((origin - test) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# Calculate mean Peak Signal-to-Noise Ratio
if __name__ == '__main__':
    # Pick random validation set
    num_subdirs = len.os.listdir(dataset+"/val")
    valdir = dataset+"/val/"+str(numpy.random.randint(0, num_subdirs-1))
    for file in os.listdir(valdir):
        images.append(Image.open(valdir+"/"+file))
    holed_images = images.copy()
    for i in range(0, len(images)):
        holed_images[i] = remove_hole_image(images[i], hole_type)
    try:
        model = load_model("unet.h5")
        print("UNet model found, starting calculations")
        generated_images = unet.predict(model, holed_images)
        psnr = 0
        for i in range(0, len(images)):
            psnr += psnr(images[i], generated_images[i])
        result = "PSNR UNet: " + str(psnr / len(images))
        print(result)
    except OSError:
        print("No UNet model found!")