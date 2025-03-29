How to use python predictor:
  1. Makesure that unetPredict.h is it the directory
  1. run > python3 RoadPredictor256x256.py 'BareEarth_Hillshade_Path' 'BareEarth_DEM_Path' 'NAIP_1m_Path'
  2. A file called 'recreatedImage.png' should be in the directory with all the data predicted

Approaches Taken
  1. Small CNN Autoencoder with only Elevation Data
  2. UNET CNN with Elevation and 
