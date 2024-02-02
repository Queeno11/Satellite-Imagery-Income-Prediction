import processing
import numpy as np
import os
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsRasterLayer,
    QgsHueSaturationFilter,
    QgsBrightnessContrastFilter,
)
from osgeo import gdal_array


# Function to calculate percentile values for each band
# def calculate_percentiles(input_path, q=3):
#     rasterArray = gdal_array.LoadFile(input_path)  # Read raster as numpy array
#     bands_percentiles = {}
#     for band in [0, 1, 2, 3]:
#         band_data = np.random.choice(rasterArray[band].flatten(), 1_000_000)
#         bands_percentiles[band] = (
#             np.percentile(band_data, q),
#             np.percentile(band_data, 100 - q),
#         )
#         print(f"{band}th band percentiles are: {bands_percentiles[band]}")

#     return bands_percentiles


def get_sample_array(input_path, sample=5_000):
    rasterArray = gdal_array.LoadFile(input_path)  # Read raster as numpy array
    bands_data = {}
    for band in [0, 1, 2, 3]:
        bands_data[band] = np.random.choice(rasterArray[band].flatten(), sample)

    return bands_data


# Parameters for the algorithm
path_in = r"D:\Maestría\Tesis\Repo\data\data_in\Pansharpened"
path_out = r"D:\Maestría\Tesis\Repo\data\data_in\Compressed\2022"

images = os.listdir(path_in)
images = [img for img in images if img.endswith(".tif")]

## Compute percentiles over all the images
# images_sample_data = {}
# for image in images:
#     # Load array
#     rasterArray = gdal_array.LoadFile(
#         f"{path_in}\{image}"
#     )  # Read raster as numpy array

#     # Get percentile values for the input image
#     images_sample_data[image] = get_sample_array(f"{path_in}\{image}")

# bands_percentiles = {}
# q = 2
# for band in range(4):
#     band_data = np.concatenate(
#         [data[band] for name, data in images_sample_data.items()]
#     )
#     bands_percentiles[band] = (
#         np.percentile(band_data, q),
#         np.percentile(band_data, 100 - q),
#     )

# # Construct the -scale options
# scale_options = " ".join(
#     [
#         f"-scale_{i+1} {min_val} {max_val}"
#         for i, (min_val, max_val) in bands_percentiles.items()
#     ]
# )
# with open(f"{path_in}/scale_options.txt", "w") as text _file:
#     text_file.write(scale_options)

## Read scale options (previous line might take a lot of time)
with open(rf"{path_in}/scale_options.txt", "r") as text_file:
    scale_options = text_file.read()


i = 0
n_imgs = len(images)
for image in images:
    ## Process the layr
    # Load layer
    file_in = f"{path_in}/{image}"
    file_out = f"{path_out}/{image}"
    layer = QgsRasterLayer(file_in)

    # Adjust gamma and brightness
    contrastFilter = QgsBrightnessContrastFilter()
    # contrastFilter.setBrightness(-20) #For brightness use contrastFilter.setBrightness(50)
    contrastFilter.setGamma(1.2)
    huesat = QgsHueSaturationFilter()
    huesat.setSaturation(30)
    layer.pipe().set(huesat)
    layer.triggerRepaint()

    layer.pipe().set(contrastFilter)
    layer.triggerRepaint()

    processing.run(
        "gdal:translate",
        {
            "INPUT": layer,
            "TARGET_CRS": QgsCoordinateReferenceSystem("EPSG:4326"),
            "NODATA": None,
            "COPY_SUBDATASETS": False,
            "OPTIONS": "COMPRESS=JPEG|JPEG_QUALITY=75|TILED=Yes",
            "EXTRA": scale_options,
            "DATA_TYPE": 1,
            "OUTPUT": file_out,
        },
    )
    assert os.path.isfile(file_out)
    print(f"{file_out} created. {n_imgs-i} images remains.")
    i += 1
