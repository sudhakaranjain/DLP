# Deep Learning Project: Image Inpainting using GANs

Rick Clement (s3852903)
Remco Pronk (s2533081)
Sudhakaran Jain (s3558487)
Jan Willem de Wit (s2616602)


To run the code, the CelebA aligned and cropped dataset is needed. It can be downloaded here: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

It needs to be placed in the same directory as the code, or you can change the path in the code. The dataset needs to be split into multiple folders first. This is done automatically when first running the program, but it will take a while depending on the speed of your computer.

We provided a requirements.txt file so you can install the packages we used.

To run the GAN:
```
python3 inpainting.py
```
To run the U-Net:
```
python3 unet.py
```

We recommend running the U-Net, as it will be a lot quicker. It will create a folder with training or testing results automatically.
