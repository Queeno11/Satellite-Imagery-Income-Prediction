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
    env = dotenv_values(r"D:/Maestría/Tesis/Repo/scripts/globals_win.env")

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
import build_dataset
import utils


def get_predicted_values(model, test_generator, kind="reg"):
    predicted_values = model.predict(test_generator)
    if kind == "cla":
        predicted_values = np.argmax(predicted_values, axis=1)
    return predicted_values


def get_real_values(bathced_dataset):
    unbathced_dataset = bathced_dataset.unbatch()
    real_values = [
        lab.numpy() for im, lab in unbathced_dataset.take(-1)
    ]  # -1 means taking every element of dataset
    return real_values


def plot_confusion_matrix(y_true, y_pred, model_name, normalize="pred"):
    from matplotlib import rcParams
    from sklearn.metrics import confusion_matrix

    # figure size in inches
    rcParams["figure.figsize"] = 11.7, 8.27

    assert (
        normalize == "pred" or normalize == "true"
    ), "normalize must be 'pred' or 'true'"
    if normalize == "pred":
        rows, cols = y_true, y_pred
        rows_lab, cols_lab = "Real Values", "Predicted Values"
    elif normalize == "true":
        rows, cols = y_pred, y_true
        rows_lab, cols_lab = "Predicted Values", "Real Values"
    # Get confusion matrix as array
    result = confusion_matrix(rows, cols, normalize="true")

    ticks = range(1, 11)
    sns_plot = sns.heatmap(result, xticklabels=ticks, yticklabels=ticks, annot=True)
    sns_plot.set(xlabel=cols_lab, ylabel=rows_lab)
    sns_plot.set_title(f"Matriz de Confusión del Modelo {model_name}", fontsize=14)

    fig = sns_plot.get_figure()
    fig.text(0.1, 0.04, "Nota: Valores normalizados respecto a las filas", fontsize=10)
    return fig


# def plot_results(model_history_small_cnn: History, mean_baseline: float):
#     """This function uses seaborn with matplotlib to plot the trainig and validation losses of both input models in an
#     sns.relplot(). The mean baseline is plotted as a horizontal red dotted line.

#     Parameters
#     ----------
#     model_history_small_cnn : History
#         keras History object of the model.fit() method.
#     model_history_eff_net : History
#         keras History object of the model.fit() method.
#     mean_baseline : float
#         Result of the get_mean_baseline() function.
#     """

#     # create a dictionary for each model history and loss type
#     dict1 = {
#         "MSE": model_history_small_cnn.history["mean_squared_error"],
#         "type": "training",
#         "model": "small_cnn",
#     }
#     dict2 = {
#         "MSE": model_history_small_cnn.history["val_mean_squared_error"],
#         "type": "validation",
#         "model": "small_cnn",
#     }

#     # convert the dicts to pd.Series and concat them to a pd.DataFrame in the long format
#     s1 = pd.DataFrame(dict1)
#     s2 = pd.DataFrame(dict2)

#     df = pd.concat([s1, s2], axis=0).reset_index()
#     grid = sns.relplot(data=df, x=df["index"], y="MSE", hue="model", col="type", kind="line", legend=False)
#     # grid.set(ylim=(20, 100))  # set the y-axis limit
#     for ax in grid.axes.flat:
#         ax.axhline(
#             y=mean_baseline, color="lightcoral", linestyle="dashed"
#         )  # add a mean baseline horizontal bar to each plot
#         ax.set(xlabel="Epoch")
#     labels = ["small_cnn", "mean_baseline"]  # custom labels for the plot

#     plt.legend(labels=labels)
#     plt.savefig("training_validation.png")
#     plt.show()


def get_gridded_images_for_grid(
    model, datasets, icpag, tiles, size, resizing_size, bias, sample, to8bit
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

    # Open grid of polygons for the corresponding parameters:
    grid = gpd.read_parquet(
        rf"{path_datain}/Grillas/grid_size{size}_tiles{tiles}.parquet"
    )

    # Iterate over the center points of each image:
    # - Start point is the center of the image (tile_size / 2, start_index)
    # - End point is the maximum possible center point (link_dataset.y.size)
    # - Step is the size of each image (tile_size)

    grid["point"] = grid.centroid
    for raster_point in grid["point"]:
        # Get data for selected point
        radio_censal = icpag.loc[icpag.contains(raster_point)]
        if radio_censal.empty:
            # El radio censal no existe, es el medio del mar...
            continue

        real_value = radio_censal["var"].values[0]
        link_name = radio_censal["link"].values[0]
        link_dataset = build_dataset.get_dataset_for_link(icpag, datasets, link_name)

        # Check if the centroid of the image is within the original polygon:
        #   - if it is, then generate the n images

        image, point, bound, tbound = utils.random_image_from_census_tract(
            link_dataset,
            icpag,
            link_name,
            start_point=raster_point,
            tiles=tiles,
            size=size,
            bias=bias,
            to8bit=to8bit,
        )

        if image is not None:
            image = utils.process_image(image, resizing_size)
            geom_bound = Polygon(bound[0])  # Create polygon of the shape of the image

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
                # batch_predictions = run_model.get_batch_predictions(model, batch_images)

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
        # "predictions": all_predictions,
        "real_value": all_real_values,
    }

    df_preds = gpd.GeoDataFrame(d, geometry=all_bounds, crs="epsg:4326")

    return df_preds
