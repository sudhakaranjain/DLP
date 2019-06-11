import numpy
import math

# Help function to calculate Peak Signal-to-Noise Ratio.
def psnr(origin, test):
    mse = numpy.mean((origin - test) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))