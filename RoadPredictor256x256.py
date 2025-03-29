import tensorflow as tf
import os
import rasterio
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import sys

def main(args):
    if(len(args) != 3):
        print('Error: Not enough args, python3 RoadPredictor256x256 BareEarth_Hillshade_Path BareEarth_DEM_Path NAIP_1m_Path')
        exit()

    model = tf.keras.models.load_model('./unetPredict.h5')

    hillshade_path = args[0]
    with rasterio.open(hillshade_path) as hillshade_dataset:
        hillshade = hillshade_dataset.read(1,masked=True)

    DEM_path = args[1]
    with rasterio.open(DEM_path) as DEM_dataset:
        DEM = DEM_dataset.read()

    NAIP_1m_path = args[2]
    with rasterio.open(NAIP_1m_path) as NAIP_1m_dataset:
        NAIP_1m = NAIP_1m_dataset.read()

    print(DEM.shape)

    xSize, ySize = hillshade.shape
    print(xSize)
    print(ySize)

    subImageCount = int(((xSize - (xSize % 256))/256) * ((ySize - (ySize % 256))/256))

    DEMSubImages = np.zeros((subImageCount, 256, 256))
    HillShadeImages = np.zeros((subImageCount, 256, 256))
    NAIP_1mSubImages1 = np.zeros((subImageCount, 256, 256))
    NAIP_1mSubImages2 = np.zeros((subImageCount, 256, 256))
    NAIP_1mSubImages3 = np.zeros((subImageCount, 256, 256))
    NAIP_1mSubImages4 = np.zeros((subImageCount, 256, 256))
    
    Index = -1
    for n in range(0, (xSize - (xSize % 256)), 256):
      for l in range(0, (ySize - (ySize % 256)), 256):
              Index = Index + 1
              DEMSubImages[Index] = DEM[0, n:(n+256), l:(l+256)]
              HillShadeImages[Index] = hillshade[n:(n+256),l:(l+256)]
              NAIP_1mSubImages1[Index] = NAIP_1m[0,n:(n+256),l:(l+256)]
              NAIP_1mSubImages2[Index] = NAIP_1m[1,n:(n+256),l:(l+256)]
              NAIP_1mSubImages3[Index] = NAIP_1m[2,n:(n+256),l:(l+256)]
              NAIP_1mSubImages4[Index] = NAIP_1m[3,n:(n+256),l:(l+256)]

    del NAIP_1m
    del DEM
    del hillshade
    
    inputImages = np.zeros((subImageCount, 256, 256, 6))
    for n in range(subImageCount):
        inputImages[n,:,:,0] = HillShadeImages[n]
        inputImages[n,:,:,1] = DEMSubImages[n]
        inputImages[n,:,:,2] = NAIP_1mSubImages1[n]
        inputImages[n,:,:,3] = NAIP_1mSubImages2[n]
        inputImages[n,:,:,4] = NAIP_1mSubImages3[n]
        inputImages[n,:,:,5] = NAIP_1mSubImages4[n]

    print('checkpoint')

    outputImages = model.predict(inputImages)

    recreatedImage = np.zeros((12800, 10240))
    Index = -1
    for n in range(0, 12800, 256):
        for l in range(0, 10240, 256):
            Index = Index + 1
            recreatedImage[n:(n+256),l:(l+256)] = outputImages[Index,:,:,0]

    plt.figure(figsize=(15, 15))
            
    plt.imshow(recreatedImage, cmap='gray', vmin=0, vmax=1)
    
    plt.savefig("recreatedImage.png")

if __name__ == "__main__":
    main(sys.argv[1:])