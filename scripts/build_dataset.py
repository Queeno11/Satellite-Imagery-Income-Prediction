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

envpath = r"/mnt/d/Maestría/Tesis/Repo/scripts/globals.env"
if os.path.isfile(envpath):
    env = dotenv_values(envpath)
else:
    env = dotenv_values(r"D:\Maestría\Tesis\Repo\scripts\globals_win.env")

path_datain = env["PATH_DATAIN"]
path_dataout = env["PATH_DATAOUT"]
path_scripts = env["PATH_SCRIPTS"]
path_satelites = env["PATH_SATELITES"]
path_logs = env["PATH_LOGS"]
path_outputs = env["PATH_OUTPUTS"]
path_imgs = env["PATH_IMGS"]
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


def load_satellite_datasets(stretch=False):
    """Load satellite datasets and get their extents"""

    files = os.listdir(rf"{path_satelites}/Pansharpened/2013")
    assert os.path.isdir(rf"{path_satelites}/Pansharpened/2013")
    files = [f for f in files if f.endswith(".tif")]
    assert all(
        [os.path.isfile(rf"{path_satelites}/Pansharpened/2013/{f}") for f in files]
    )

    datasets = {
        f.replace(".tif", ""): (
            xr.open_dataset(rf"{path_satelites}/Pansharpened/2013/{f}")
        )
        for f in files
    }
    if stretch:
        datasets = {name:stretch_dataset(ds) for name, ds in datasets.items()}
        
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


def assign_links_to_datasets(icpag, extents, verbose=True):
    """Assign each link a dataset if the census tract falls within the extent of the dataset (images)"""
    import warnings

    warnings.filterwarnings("ignore")

    for name, bbox in extents.items():
        icpag.loc[icpag.within(bbox), "dataset"] = name

    nan_links = icpag.dataset.isna().sum()
    icpag = icpag[icpag.dataset.notna()]

    if verbose:
        print("Links without images:", nan_links, "out of", len(icpag))
        icpag.plot()
        plt.savefig(rf"{path_dataout}/links_with_images.png")

    warnings.filterwarnings("default")

    return icpag


def split_train_test(df):
    """Blocks are counted from left to right, one count for test and one for train."""

    # Set bounds of test dataset blocks
    test1_max_x = -58.66
    test1_min_x = -58.71
    test2_max_x = -58.36
    test2_min_x = -58.41

    # These blocks are the test dataset.
    #   The hole image have to be inside the train region
    df["type"] = np.nan
    df.loc[
        (
            (df.min_x > test1_min_x) & (df.max_x < test1_max_x)
        )  # Entre test1_minx y test1_maxx
        | (
            (df.min_x > test2_min_x) & (df.max_x < test2_max_x)
        ),  # Entre test2_minx y test2_maxx
        "type",
    ] = "test"

    # These blocks are the train dataset.
    #   The hole image have to be inside the train region
    df.loc[df.max_x < test1_min_x, "train_block"] = 1  # A la izqauierda de test1
    df.loc[
        (df.min_x > test1_max_x) & (df.max_x < test2_min_x), "train_block"
    ] = 2  # Entre test1 y test2
    df.loc[df.min_x > test2_max_x, "train_block"] = 3  # A la derecha de test2
    # print(df.train_block.value_counts())

    # Put nans in the overlapping borders
    df.loc[df.train_block.isin([1, 2, 3]), "type"] = "train"
    df = df.drop(columns="train_block")

    test_size = df[df["type"] == "test"].shape[0]
    train_size = df[df["type"] == "train"].shape[0]
    invalid_size = df[df["type"].isna()].shape[0]
    total_size = df["type"].shape[0]

    print(
        "",
        f"Size of test dataset: {test_size/total_size*100:.2f}% ({test_size} census tracts)",
        f"Size of train dataset: {train_size/total_size*100:.2f}% ({train_size} census tracts)",
        f"Deleted images due to train/test overlapping: {invalid_size/total_size*100:.2f}% ({invalid_size} census tracts))",
        sep="\n",
    )

    return df

def assert_train_test_datapoint(bounds, wanted_type="train"):
    min_x, _, max_x, _ = bounds  # Ignore min_y and max_y

    test_blocks = [(-58.71, -58.66), (-58.41, -58.36)]

    for (test_min_x, test_max_x) in test_blocks:
        if test_min_x < min_x < max_x < test_max_x:
            # Inside test bloc
            return wanted_type == "test"
        elif max_x < test_min_x:
            return wanted_type == "train"
        elif test_max_x < min_x < max_x:
            return wanted_type == "train"
        elif max_x < test_max_x:
            return wanted_type == "train"

    return False


def assert_train_test_datapoint(bounds, wanted_type="train"):
    """Returns True if the datapoint is of the wanted type (train or test)."""

    # Split bounds:
    min_x, _, max_x, _ = bounds
    # min_x = bounds[0]
    # max_x = bounds[1]

    # Set bounds of test dataset blocks
    #    Note: Blocks are counted from left to right, one count for test and one for train.
    test1_max_x = -58.66
    test1_min_x = -58.71
    test2_max_x = -58.36
    test2_min_x = -58.41

    # These blocks are the test dataset.
    #   All the images have to be inside the test dataset blocks,
    #   so the filter is based on x_min and x_max of the images.
    type = None
    if ((min_x > test1_min_x) & (max_x < test1_max_x)) | (
        (min_x > test2_min_x) & (max_x < test2_max_x)
    ):
        type = "test"

    # These blocks are the train dataset.
    #   The hole image have to be inside the train region
    if max_x < test1_min_x:
        train_block = 1
    elif (min_x > test1_max_x) & (max_x < test2_min_x):
        train_block = 2
    elif min_x > test2_max_x:
        train_block = 3
    else:
        train_block = None

    # Put nans in the overlapping borders
    if (train_block == 1) | (train_block == 2) | (train_block == 3):
        type = "train"

    # Assert type
    if type == wanted_type:
        istype = True
    else:
        istype = False

    return istype


def get_dataset_for_link(icpag, datasets, link):
    """Get dataset where the census tract is located.

    Parameters
    ----------
    - icpag: geopandas.GeoDataFrame, shapefile with the census tracts
    - datasets: dict, dictionary with the satellite datasets. Names of the datasets are the keys and xarray.Datasets are the values.
    - link: str, 9 digits that identify the census tract

    Returns
    -------
    - current_ds: xarray.Dataset, dataset where the census tract is located
    """
    current_ds_name = icpag.loc[icpag.link == link, "dataset"].values[0]
    current_ds = datasets[current_ds_name]
    return current_ds


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
    train_path = rf"{path_imgs}/train_size{image_size}_tiles{tiles}_sample{sample_size}"
    os.makedirs(train_path, exist_ok=True)

    # Generate images
    print("Generando imágenes...")
    metadata = {}
    actual_size = image_size // tiles * tiles
    for link in tqdm(icpag.link.unique()):
        link_dataset = get_dataset_for_link(icpag, datasets, link)

        for n in range(sample_size):
            # FIXME: algo se rompe con n=2... por qué?
            name = f"{link}_{n}"

            img = np.zeros((4, 2, 2))

            path_image = rf"{train_path}/{name}.npy"

            # The image could be in the border of the dataset, so we need to try again until we get a valid image
            img, point, bounds, total_bounds = utils.random_image_from_census_tract(
                link_dataset, icpag, link, tiles=tiles, size=actual_size, bias=bias
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
    path_metadata = rf"{train_path}/metadata.csv"
    metadata.to_csv(path_metadata)
    print(
        f"Se creó {path_metadata} y las imágenes para correr el modelo: train_size{image_size}_tiles{tiles}_sample{sample_size}."
    )


def crop_dataset_to_link(ds, icpag, link):
    # obtengo el poligono correspondiente al link
    gdf_slice = icpag.loc[icpag["link"] == link]
    # Get bounds of the shapefile's polygon
    bbox_img = gdf_slice.bounds

    # Filter dataset based on the bounds of the shapefile's polygon
    image_ds = ds.sel(
        x=slice(float(bbox_img["minx"]), float(bbox_img["maxx"])),
        y=slice(float(bbox_img["maxy"]), float(bbox_img["miny"])),
    )
    return image_ds


def get_gridded_images_for_link(
    ds, icpag, link, tiles, size, resizing_size, bias, sample, to8bit, n_bands=4, stacked_images=[1],
):
    """
    Itera sobre el bounding box del poligono del radio censal, tomando imagenes de tamño sizexsize
    Si dicha imagen se encuentra dentro del polinogo, se genera el composite con dicha imagen mas otras tiles**2 -1 imagenes
    Devuelve un array con todas las imagenes generadas, un array con los puntos centrales de cada imagen y un array con los bounding boxes de cada imagen.

    Parameters:
    -----------
    ds: xarray.Dataset, dataset con las imágenes de satélite
    icpag: geopandas.GeoDataFrame, shapefile con los radios censales
    link: str, 9 dígitos que identifican el radio censal
    tiles: int, cantidad de imágenes a generar por lado
    size: int, tamaño de la imagen a generar, en píxeles
    resizing_size: int, tamaño al que se redimensiona la imagen
    bias: int, cantidad de píxeles que se mueve el punto aleatorio de las tiles
    sample: int, cantidad de imágenes a generar por box (util cuando tiles > 1)
    to8bit: bool, si es True, convierte la imagen a 8 bits

    Returns:
    --------
    images: list, lista con las imágenes generadas
    points: list, lista con los puntos centrales de cada imagen
    bounds: list, lista con los bounding boxes de cada imagen
    """
    # FIXME: algunos radios censales no se generan bien. Ejemplo: 065150101. ¿Que pasa ahi?
    images = []
    points = []
    bounds = []
    tile_size = size // tiles
    tiles_generated = 0

    link_dataset = crop_dataset_to_link(ds, icpag, link)
    # FIXME: add margin to the bounding box so left and bottom tiles are not cut. Margin should be the size of the tile - 1
    link_geometry = icpag.loc[icpag["link"] == link, "geometry"].values[0]

    # Iterate over the center points of each image:
    # - Start point is the center of the image (tile_size / 2, start_index)
    # - End point is the maximum possible center point (link_dataset.y.size)
    # - Step is the size of each image (tile_size)
    start_index = int(tile_size / 2)
    for idy in range(start_index, link_dataset.y.size, tile_size):
        # Iterate over columns
        for idx in range(start_index, link_dataset.x.size, tile_size):
            # Get the center point of the image
            image_point = (float(link_dataset.x[idx]), float(link_dataset.y[idy]))
            point_geom = sg.Point(image_point)

            # Check if the centroid of the image is within the original polygon:
            #   - if it is, then generate the n images
            if link_geometry.contains(point_geom):  # or intersects
                number_imgs = 0
                counter = 0  # Limit the times to try to sample the images
                while (number_imgs < sample) & (counter < sample * 2):
                    polygon = icpag.at[icpag["link"]==link, "geometry"]
                    img, bound = utils.stacked_image_from_census_tract(
                        dataset=ds,
                        polygon=polygon,
                        img_size=size,
                        n_bands=n_bands,
                        stacked_images=stacked_images
                    )

                    counter += 1
                    print(counter)

                    if img is not None:
                        # TODO: add a check to see if the image is contained in test bounds
                        img = utils.process_image(img, resizing_size)

                        images += [img]
                        bounds += [bound]
                        number_imgs += 1

                    else:
                        print("Image failed")

    return images, points, bounds


def get_gridded_images_for_dataset(
    model, ds, icpag, tiles, size, resizing_size, bias, sample, to8bit
):
    """
    Itera sobre el bounding box de un dataset (raster de imagenes), tomando imagenes de tamño sizexsize
    Asigna el valor "real" del radio censal al que pertenece el centroide de la imagen.
    Devuelve un array con todas las imagenes generadas, un array con los puntos centrales de cada imagen,
    un array con los valores "reales" de los radios censales y un array con los bounding boxes de cada imagen.

    Parameters:
    -----------
    ds: xarray.Dataset, dataset con las imágenes de satélite
    icpag: geopandas.GeoDataFrame, shapefile con los radios censales
    tiles: int, cantidad de imágenes a generar por lado
    size: int, tamaño de la imagen a generar, en píxeles
    resizing_size: int, tamaño al que se redimensiona la imagen
    bias: int, cantidad de píxeles que se mueve el punto aleatorio de las tiles
    sample: int, cantidad de imágenes a generar por box (util cuando tiles > 1)
    to8bit: bool, si es True, convierte la imagen a 8 bits

    Returns:
    --------
    images: list, lista con las imágenes generadas
    points: list, lista con los puntos centrales de cada imagen
    bounds: list, lista con los bounding boxes de cada imagen
    """
    import run_model
    from shapely.geometry import Polygon

    # FIXME: algunos radios censales no se generan bien. Ejemplo: 065150101. ¿Que pasa ahi?
    # Inicializo arrays
    batch_images = np.empty((0, resizing_size, resizing_size, 4))
    batch_link_names = np.empty((0))
    batch_predictions = np.empty((0))
    batch_real_values = np.empty((0))
    batch_bounds = np.empty((0))
    all_link_names = np.empty((0))
    all_predictions = np.empty((0))
    all_real_values = np.empty((0))
    all_bounds = np.empty((0))

    tile_size = size // tiles
    tiles_generated = 0

    # Iterate over the center points of each image:
    # - Start point is the center of the image (tile_size / 2, start_index)
    # - End point is the maximum possible center point (link_dataset.y.size)
    # - Step is the size of each image (tile_size)

    # FIXME: para mejorar la eficiencia, convendría hacer un dissolve de icpag y verificar que
    # point_geom este en ese polygono y no en todo el df
    start_index = int(tile_size / 2)
    for idy in range(start_index, ds.y.size, tile_size):
        # Iterate over columns
        for idx in range(start_index, ds.x.size, tile_size):
            # Get the center point of the image
            image_point = (float(ds.x[idx]), float(ds.y[idy]))
            point_geom = sg.Point(image_point)

            # Get data for selected point
            radio_censal = icpag.loc[icpag.contains(point_geom)]
            if radio_censal.empty:
                # El radio censal no existe, es el medio del mar...
                continue

            real_value = radio_censal["var"].values[0]
            link_name = radio_censal["link"].values[0]

            # Check if the centroid of the image is within the original polygon:
            #   - if it is, then generate the n images

            image, point, bound, tbound = utils.random_image_from_census_tract(
                ds,
                icpag,
                link_name,
                start_point=image_point,
                tiles=tiles,
                size=size,
                bias=bias,
                to8bit=to8bit,
            )

            if image is not None:
                image = utils.process_image(image, resizing_size)
                geom_bound = Polygon(
                    bound[0]
                )  # Create polygon of the shape of the image

                batch_images = np.concatenate([batch_images, np.array([image])], axis=0)
                batch_link_names = np.concatenate(
                    [batch_link_names, np.array([link_name])], axis=0
                )
                batch_real_values = np.concatenate(
                    [batch_real_values, np.array([real_value])], axis=0
                )
                batch_bounds = np.concatenate(
                    [batch_bounds, np.array([geom_bound])], axis=0
                )

                # predict with the model over the batch
                if batch_images.shape[0] == 128:
                    # predictions
                    batch_predictions = run_model.get_batch_predictions(
                        model, batch_images
                    )

                    # Store data
                    all_predictions = np.concatenate(
                        [all_predictions, batch_predictions], axis=0
                    )
                    all_link_names = np.concatenate(
                        [all_link_names, batch_link_names], axis=0
                    )
                    all_real_values = np.concatenate(
                        [all_real_values, batch_real_values], axis=0
                    )
                    all_bounds = np.concatenate([all_bounds, batch_bounds], axis=0)

                    # Restore batches to empty
                    batch_images = np.empty((0, resizing_size, resizing_size, 4))
                    batch_predictions = np.empty((0))
                    batch_link_names = np.empty((0))
                    batch_predictions = np.empty((0))
                    batch_real_values = np.empty((0))
                    batch_bounds = np.empty((0))

    # Creo dataframe para exportar:
    d = {
        "link": all_link_names,
        "predictions": all_predictions,
        "real_value": all_real_values,
    }

    df_preds = gpd.GeoDataFrame(d, geometry=all_bounds, crs="epsg:4326")

    return df_preds


def get_gridded_images_for_grid(
    model, ds, icpag, tiles, size, resizing_size, bias, sample, to8bit
):
    """
    Itera sobre el bounding box de un dataset (raster de imagenes), tomando imagenes de tamño sizexsize
    Asigna el valor "real" del radio censal al que pertenece el centroide de la imagen.
    Devuelve un array con todas las imagenes generadas, un array con los puntos centrales de cada imagen,
    un array con los valores "reales" de los radios censales y un array con los bounding boxes de cada imagen.

    Parameters:
    -----------
    ds: xarray.Dataset, dataset con las imágenes de satélite
    icpag: geopandas.GeoDataFrame, shapefile con los radios censales
    tiles: int, cantidad de imágenes a generar por lado
    size: int, tamaño de la imagen a generar, en píxeles
    resizing_size: int, tamaño al que se redimensiona la imagen
    bias: int, cantidad de píxeles que se mueve el punto aleatorio de las tiles
    sample: int, cantidad de imágenes a generar por box (util cuando tiles > 1)
    to8bit: bool, si es True, convierte la imagen a 8 bits

    Returns:
    --------
    images: list, lista con las imágenes generadas
    points: list, lista con los puntos centrales de cada imagen
    bounds: list, lista con los bounding boxes de cada imagen
    """
    import run_model
    from shapely.geometry import Polygon

    # FIXME: algunos radios censales no se generan bien. Ejemplo: 065150101. ¿Que pasa ahi?
    # Inicializo arrays
    batch_images = np.empty((0, resizing_size, resizing_size, 4))
    batch_link_names = np.empty((0))
    batch_predictions = np.empty((0))
    batch_real_values = np.empty((0))
    batch_bounds = np.empty((0))
    all_link_names = np.empty((0))
    all_predictions = np.empty((0))
    all_real_values = np.empty((0))
    all_bounds = np.empty((0))

    tile_size = size // tiles
    tiles_generated = 0

    # Iterate over the center points of each image:
    # - Start point is the center of the image (tile_size / 2, start_index)
    # - End point is the maximum possible center point (link_dataset.y.size)
    # - Step is the size of each image (tile_size)

    # FIXME: para mejorar la eficiencia, convendría hacer un dissolve de icpag y verificar que
    # point_geom este en ese polygono y no en todo el df
    start_index = int(tile_size / 2)
    for idy in range(start_index, ds.y.size, tile_size):
        # Iterate over columns
        for idx in range(start_index, ds.x.size, tile_size):
            # Get the center point of the image
            image_point = (float(ds.x[idx]), float(ds.y[idy]))
            point_geom = sg.Point(image_point)

            # Get data for selected point
            radio_censal = icpag.loc[icpag.contains(point_geom)]
            if radio_censal.empty:
                # El radio censal no existe, es el medio del mar...
                continue

            real_value = radio_censal["var"].values[0]
            link_name = radio_censal["link"].values[0]

            # Check if the centroid of the image is within the original polygon:
            #   - if it is, then generate the n images

            image, point, bound, tbound = utils.random_image_from_census_tract(
                ds,
                icpag,
                link_name,
                start_point=image_point,
                tiles=tiles,
                size=size,
                bias=bias,
                to8bit=to8bit,
            )

            if image is not None:
                image = utils.process_image(image, resizing_size)
                geom_bound = Polygon(
                    bound[0]
                )  # Create polygon of the shape of the image

                batch_images = np.concatenate([batch_images, np.array([image])], axis=0)
                batch_link_names = np.concatenate(
                    [batch_link_names, np.array([link_name])], axis=0
                )
                batch_real_values = np.concatenate(
                    [batch_real_values, np.array([real_value])], axis=0
                )
                batch_bounds = np.concatenate(
                    [batch_bounds, np.array([geom_bound])], axis=0
                )

                # predict with the model over the batch
                if batch_images.shape[0] == 128:
                    # predictions
                    batch_predictions = run_model.get_batch_predictions(
                        model, batch_images
                    )

                    # Store data
                    all_predictions = np.concatenate(
                        [all_predictions, batch_predictions], axis=0
                    )
                    all_link_names = np.concatenate(
                        [all_link_names, batch_link_names], axis=0
                    )
                    all_real_values = np.concatenate(
                        [all_real_values, batch_real_values], axis=0
                    )
                    all_bounds = np.concatenate([all_bounds, batch_bounds], axis=0)

                    # Restore batches to empty
                    batch_images = np.empty((0, resizing_size, resizing_size, 4))
                    batch_predictions = np.empty((0))
                    batch_link_names = np.empty((0))
                    batch_predictions = np.empty((0))
                    batch_real_values = np.empty((0))
                    batch_bounds = np.empty((0))

    # Creo dataframe para exportar:
    d = {
        "link": all_link_names,
        "predictions": all_predictions,
        "real_value": all_real_values,
    }

    df_preds = gpd.GeoDataFrame(d, geometry=all_bounds, crs="epsg:4326")

    return df_preds


def get_random_images_for_link(
    ds, icpag, link, tiles, size, resizing_size, bias, sample, to8bit
):
    """
    Genera n imagenes del poligono del radio censal, tomando imagenes de tamño sizexsize
    Si dicha imagen se encuentra dentro del polinogo, se genera el composite con dicha imagen mas otras tiles**2 -1 imagenes
    Devuelve un array con todas las imagenes generadas, un array con los puntos centrales de cada imagen y un array con los bounding boxes de cada imagen.

    Parameters:
    -----------
    ds: xarray.Dataset, dataset con las imágenes de satélite
    icpag: geopandas.GeoDataFrame, shapefile con los radios censales
    link: str, 9 dígitos que identifican el radio censal
    tiles: int, cantidad de imágenes a generar por lado
    size: int, tamaño de la imagen a generar, en píxeles
    resizing_size: int, tamaño al que se redimensiona la imagen
    bias: int, cantidad de píxeles que se mueve el punto aleatorio de las tiles
    sample: int, cantidad de imágenes a generar por box (util cuando tiles > 1)
    to8bit: bool, si es True, convierte la imagen a 8 bits

    Returns:
    --------
    images: list, lista con las imágenes generadas
    points: list, lista con los puntos centrales de cada imagen
    bounds: list, lista con los bounding boxes de cada imagen
    """
    # FIXME: algunos radios censales no se generan bien. Ejemplo: 065150101. ¿Que pasa ahi?
    images = []
    points = []
    bounds = []
    tile_size = size // tiles
    tiles_generated = 0

    link_dataset = crop_dataset_to_link(ds, icpag, link)
    # FIXME: add margin to the bounding box so left and bottom tiles are not cut. Margin should be the size of the tile - 1
    link_geometry = icpag.loc[icpag["link"] == link, "geometry"].values[0]

    number_imgs = 0
    counter = 0  # Limit the times to try to sample the images
    while (number_imgs < sample) & (counter < sample * 2):
        # Generate a random point
        x_point = np.random.uniform(link_dataset.x.min(), link_dataset.x.max())
        y_point = np.random.uniform(link_dataset.y.min(), link_dataset.y.max())

        # Get the center point of the image
        image_point = (x_point, y_point)
        point_geom = sg.Point(image_point)

        # Check if the centroid of the image is within the original polygon:
        #   - if it is, then generate the n images
        if link_geometry.contains(point_geom):  # or intersects
            img, point, bound, tbound = utils.random_image_from_census_tract(
                ds,
                icpag,
                link,
                start_point=image_point,
                tiles=tiles,
                size=size,
                bias=bias,
                to8bit=to8bit,
            )

            counter += 1
            print(counter)

            if img is not None:
                # TODO: add a check to see if the image is contained in test bounds
                img = utils.process_image(img, resizing_size)

                images += [img]
                points += [point]
                bounds += [bound]
                number_imgs += 1

            else:
                print("Image failed")

    return images, points, bounds

    return images, real_values, links, points, bounds

def stretch_dataset(ds, pixel_depth=32_767):
    ''' Stretch band data from satellite images. '''
    minimum = ds.band_data.quantile(.01).values
    maximum = ds.band_data.quantile(.99).values
    ds = (ds - minimum) / (maximum - minimum) * pixel_depth
    ds = ds.where(ds.band_data > 0, 0)
    ds = ds.where(ds.band_data < pixel_depth, pixel_depth)
    return ds


if __name__ == "__main__":
    # Parameters
    image_size = 512
    sample_size = 10

    # Generate dataset
    build_dataset(image_size, sample_size, variable="pred_inc_mean")
