# Team 11

## Team Members
| Name             | Email                          | GitHub Username   |
|-----------------|--------------------------------|------------------|
| Tatiana Gabel  | Tatiana.Gabel@colostate.edu    | TianaGabel      |
| Reed Johns     | Reed.Johns@colostate.edu       | vertumnal       |
| Lennox Nutall  | Lennox.Nutall@colostate.edu    | LennoxNutall    |
| Sawyer Jacobson| Sawyer.Jacobson@colostate.edu  | sawyerjacobson  |
| Gavin Hawkes   | Gavin.Hawkes@colostate.edu     | -               |

## Project Links
- **Presentation:** [Link to Presentation](https://colostate-my.sharepoint.com/:p:/g/personal/c837386285_colostate_edu/EZeOLh1XLapDqXTXQy-v304Br2KbG-2_lEEgCVxQoCL_sw?e=xxb1gA)

## Description of Notebooks

### Preprocessing
- **Patching**: Processes large geospatial datasets by dividing NAIP imagery, road boundary masks, hillshade, and DEM data into 512x512 pixel patches while preserving metadata and georeferencing. It filters patches based on a shapefile boundary, removes those outside the area of interest, and randomly selects 20% of the remaining patches for the test dataset.
- **normalizeWholeImage**: Normalizes the DEM image.
- **RasterBlaster**: Acts as the main function to run other raster operations and patching scripts.
- **RasterSlasher**: Automates the patching process.
- **RasterSmasher**: Stacks patches from NAIP imagery, road boundary masks, hillshade, and DEM data, generating an 8-band TIFF file.

### Model
- **model-super-epoch**: Our most recent model, using a U-Net architecture.
- **model-previous**: The initial model designed to work with RGB data.
- **model**: Contains both model training and inference.
  - *One cell*: Runs the final U-Net model.
  - *A later cell*: Processes test images and generates predicted masks.

### PostProcessing
- **maskStitcher**: Recreates a single binary mask TIFF from outputted patches, preserving geo metadata.
- **postImageProcessing**: Processes a raster file by converting it to a binary black-and-white image, applying dilation and thinning using the Zhang-Suen algorithm, and extracting the skeleton of the structure. This could potentially connect road sections that appear disconnected but likely are.

### Miscellaneous
- **outputStats**: Calculates basic statistics to assess output vs. ground truth.
- **pca**: Runs a Principal Component Analysis (PCA) to evaluate the contribution of each band to data variance.

