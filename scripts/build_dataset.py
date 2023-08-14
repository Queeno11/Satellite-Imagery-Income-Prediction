##############      Configuración      ##############
import os
import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from PIL import Image
from dotenv import dotenv_values

pd.set_option("display.max_columns", None)
# env = dotenv_values("/mnt/d/Maestría/Tesis/Repo/scripts/globals.env")
env = dotenv_values(r"D:/Maestría/Tesis/Repo/scripts/globals.env")

path_proyecto = env["PATH_PROYECTO"]
path_datain = env["PATH_DATAIN"]
path_dataout = env["PATH_DATAOUT"]
path_scripts = env["PATH_SCRIPTS"]
path_satelites = env["PATH_SATELITES"]
path_logs = env["PATH_LOGS"]
path_outputs = env["PATH_OUTPUTS"]
# path_programas  = globales[7]

import affine
import geopandas as gpd
import rasterio.features
import xarray as xr
import rioxarray as xrx
import shapely.geometry as sg
import pandas as pd
from tqdm import tqdm
import utils


def load_satellite_datasets():
    """Load satellite datasets and get their extents"""

    files = os.listdir(rf"{path_datain}/Pansharpened/2013")
    assert os.path.isdir(rf"{path_datain}/Pansharpened/2013")
    files = [f for f in files if f.endswith(".tif")]
    assert all([os.path.isfile(rf"{path_datain}/Pansharpened/2013/{f}") for f in files])

    datasets = {
        f.replace(".tif", ""): xr.open_dataset(rf"{path_datain}/Pansharpened/2013/{f}")
        for f in files
    }
    extents = {name: utils.get_dataset_extent(ds) for name, ds in datasets.items()}

    return datasets, extents


def load_icpag_dataset(variable="ln_pred_inc_mean"):
    """Open ICPAG dataset and merge with ELL estimation."""

    # Open ICPAG dataset
    icpag = gpd.read_file(rf"{path_datain}/ICPAG/base_icpag_500k.shp")
    icpag = icpag.to_crs(epsg=4326)
    icpag = icpag[icpag.AMBA_legal == 1].reset_index(drop=True)

    # Open ELL estimation
    collapse_link = pd.read_stata(rf"{path_datain}/predict_ingreso_collapse.dta")

    # Merge icpag indicators with ELL estimation
    icpag["link"] = icpag["link"].astype(str).str.zfill(9)
    collapse_link["link"] = collapse_link["link"].astype(str).str.zfill(9)
    icpag = icpag.merge(collapse_link, on="link", how="left", validate="1:1")

    # Normalize ELL estimation:
    icpag["var"] = (icpag[variable] - icpag[variable].mean()) / icpag[variable].std()

    return icpag


def assign_links_to_datasets(icpag, extents):
    """Assign each link a dataset if the census tract falls within the extent of the dataset (images)"""

    for name, bbox in extents.items():
        icpag.loc[icpag.centroid.within(bbox), "dataset"] = name

    print("Links without images:", icpag.dataset.isna().sum(), "out of", len(icpag))

    icpag = icpag[icpag.dataset.notna()]

    icpag.plot()
    plt.savefig(rf"{path_dataout}/links_with_images.png")

    return icpag


def split_train_test(metadata):
    """Blocks are counted from left to right, one count for test and one for train."""

    # Set bounds of test dataset blocks
    test1_max_x = -58.66
    test1_min_x = -58.71
    test2_max_x = -58.36
    test2_min_x = -58.41

    # These blocks are the test dataset.
    #   All the images have to be inside the test dataset blocks,
    #   so the filter is based on x_min and x_max of the images.
    metadata["type"] = np.nan
    metadata.loc[
        ((metadata.min_x > test1_min_x) & (metadata.max_x < test1_max_x))
        | ((metadata.min_x > test2_min_x) & (metadata.max_x < test2_max_x)),
        "type",
    ] = "test"

    ## Clean overlapping borders
    # Get bounds of train dataset blocks
    metadata.loc[metadata.x < test1_min_x, "train_block"] = 1
    metadata.loc[
        (metadata.x > test1_max_x) & (metadata.x < test2_min_x), "train_block"
    ] = 2
    metadata.loc[metadata.x > test2_max_x, "train_block"] = 3
    print(metadata.train_block.value_counts())

    # Put nans in the overlapping borders
    metadata.loc[
        ((metadata.train_block == 1) & (metadata.max_x < test1_min_x))
        | (
            (metadata.train_block == 2)
            & (metadata.min_x > test1_max_x)
            & (metadata.max_x < test2_min_x)
        )
        | ((metadata.train_block == 3) & (metadata.min_x > test2_max_x)),
        "type",
    ] = "train"
    metadata = metadata.drop(columns="train_block")

    test_size = metadata[metadata.type == "test"].shape[0] / metadata.shape[0] * 100
    train_size = metadata[metadata.type == "train"].shape[0] / metadata.shape[0] * 100
    invalid_size = metadata[metadata.type.isna()].shape[0] / metadata.shape[0] * 100
    print(
        f"Size of test dataset: {test_size:.2f}%\n",
        f"Size of train dataset: {train_size:.2f}%\n",
        f"Deleted images due to train/test overlapping: {invalid_size:.2f}%\n",
    )

    return metadata


def build_dataset(
    image_size, sample_size, tiles=1, bias=2, variable="ln_pred_inc_mean"
):
    """Build dataset for training the model.

    Generates images of size (4, image_size, image_size) and saves them in a folder in npy format.

    Parameters
    ----------
    - image_size: int, size of the size of the image in pixels
    - sample_size: int, number of images to generate per census tract
    - tiles: int, number of tiles to generate per side of the image. For example, tiles = 2 will generate 4 tiles per image.
    - variable: str, variable to predict. Has to be a variable in the ICPAG dataset.
    """
    # Load datasets
    print("Cargando datasets...")
    datasets, extents = load_satellite_datasets()
    icpag = load_icpag_dataset(variable)
    icpag = assign_links_to_datasets(icpag, extents)

    # Create output folders
    os.makedirs(rf"{path_dataout}/size{image_size}_sample{sample_size}", exist_ok=True)

    # Generate images
    print("Generando imágenes...")
    metadata = {}
    actual_size = image_size // tiles * tiles
    for link in tqdm(icpag.link.unique()):
        current_ds_name = icpag.loc[icpag.link == link, "dataset"].values[0]
        current_ds = datasets[current_ds_name]

        for n in range(sample_size):
            # FIXME: algo se rompe con n=2... por qué?
            name = f"{link}_{n}"

            img = np.zeros((4, 2, 2))

            path_image = (
                rf"{path_dataout}/size{image_size}_sample{sample_size}/{name}.npy"
            )

            # The image could be in the border of the dataset, so we need to try again until we get a valid image
            img, point, bounds, total_bounds = utils.random_image_from_census_tract(
                current_ds, icpag, link, tiles=tiles, size=actual_size, bias=bias
            )

            metadata[name] = {
                "link": link,
                "sample": n,
                "image": path_image,
                "var": icpag.loc[icpag.link == link, "var"].values[0],
                "point": point,
                "x": point[0],
                "y": point[1],
                "tiles_boundaries": bounds,
                "min_x": total_bounds[0],
                "max_x": total_bounds[1],
                "min_y": total_bounds[2],
                "max_y": total_bounds[3],
            }

            np.save(
                path_image,
                img,
            )

    # Metadata to clean dataframe
    metadata = pd.DataFrame().from_dict(metadata, orient="index")
    metadata = metadata.dropna(how="any").reset_index(drop=True)

    # Train and test split
    metadata = split_train_test(metadata)

    # Export
    path_metadata = rf"{path_dataout}/size{image_size}_sample{sample_size}/metadata.csv"
    metadata.to_csv(path_metadata)
    print(
        f"Se creó {path_metadata} y las imágenes para correr el modelo: size{image_size}_sample{sample_size}."
    )


if __name__ == "__main__":
    # Parameters
    image_size = 512
    sample_size = 10

    # Generate dataset
    build_dataset(image_size, sample_size, variable="pred_inc_mean")
