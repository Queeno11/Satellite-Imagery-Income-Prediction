##############      Configuración      ##############
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from dotenv import dotenv_values

pd.set_option("display.max_columns", None)
envpath = "/mnt/d/Maestría/Tesis/Repo/scripts/globals.env"
if os.path.isfile(envpath):
    env = dotenv_values(envpath)
else:
    env = dotenv_values(r"D:/Maestría/Tesis/Repo/scripts/globals_win.env")

path_proyecto = env["PATH_PROYECTO"]
path_datain = env["PATH_DATAIN"]
path_dataout = env["PATH_DATAOUT"]
path_scripts = env["PATH_SCRIPTS"]
path_satelites = env["PATH_SATELITES"]
path_logs = env["PATH_LOGS"]
path_outputs = env["PATH_OUTPUTS"]
path_imgs = env["PATH_IMGS"]
# path_programas  = globales[7]
###############################################
import prediction_tools
import custom_models
import build_dataset
import utils

import sys
import pandas as pd
import os
import xarray as xr
from typing import Iterator, List, Union, Tuple, Any
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import tensorflow_addons as tfa

from tensorflow.keras import layers, models, Model
from tensorflow.keras.callbacks import (
    TensorBoard,
    EarlyStopping,
    ModelCheckpoint,
)
from tensorflow.keras.models import Sequential
import cv2
import skimage

# the next 3 lines of code are for my machine and setup due to https://github.com/tensorflow/tensorflow/issues/43174
try:
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print("No GPU set. Is the GPU already initialized?")


# Disable
def blockPrint():
    sys.__stdout__ = sys.stdout
    sys.stdout = open(os.devnull, "w")

# Restore
def enablePrint():
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    
    
def generate_gridded_images(
    df_test,
    sat_img_datasets,
    test_folder,
    tiles,
    size,
    resizing_size,
    n_bands,
    stacked_images
):
    import geopandas as gpd
       
    # Filtro Radios demasiado grandes (tardan horas en generar la cuadrícula y es puro campo...)
    df_test = df_test[df_test["AREA"] <= 200000]  # Remove rc that are too big
    links = df_test["link"].unique()
    len_links = len(links)
    valid_links = []

    # Loop por radio censal. Si está la imagen la usa, sino la genera.
    os.makedirs(test_folder, exist_ok=True)
    for n, link in enumerate(links):
        print(f"{link}: {n}/{len_links}")
        # Genera la imagen
        file = rf"{test_folder}/test_{link}.npy"
        link_dataset = build_dataset.get_dataset_for_link(
            df_test, sat_img_datasets, link
        )
        images, points, bounds = build_dataset.get_gridded_images_for_link(
            link_dataset,
            df_test,
            link,
            tiles,
            size,
            resizing_size,
            sample=1,
            n_bands=n_bands,
            stacked_images=stacked_images
        )
        print("imagen generada")
        if len(images) == 0:
            # No images where returned from this census tract, so no error to compute...
            print(f"problema con link {link}...")
        else:
            processed_images = []
            for image in images:
                # image = utils.process_image(image, resizing_size=resizing_size, moveaxis=False)
                processed_images += [image]
            processed_images = np.array(processed_images)
            np.save(file, processed_images)
            valid_links += [link]

    valid_links = np.array(valid_links)
    np.save(rf"{test_folder}/valid_links.npy", valid_links)
    print("Imagenes generadas!")
    
    return test_folder

def get_batch_predictions(model, batch_images):
    """Computa las predicciones del batch

    Parameters:
    -----------
    model: tf.keras.Model, modelo entrenado
    images: np.array con las imágenes a predecir (batch_size, img_size, img_size, bands)
    batch_real_value: np.array con el valor real del radio censal correspondiente (batch_size)
    metric: function, función que computa el error de predicción. Por defecto es np.mean
    """
    to_predict = tf.data.Dataset.from_tensor_slices(batch_images)

    to_predict = to_predict.batch(400)
    predictions = model.predict(to_predict)
    predictions = predictions.flatten()

    return predictions


def compute_custom_loss(
    model,
    df_test,
    sat_img_datasets,
    tiles,
    size,
    resizing_size,
    bias,
    sample,
    to8bit,
    verbose=False,
    trim_size=True,
    n_bands=4,
    stacked_images=[1]
):
    """
    Calcula el ECM del conjunto de predicción.

    Carga las bases necesarias, itera sobre los radios censales, genera las imágenes
    en forma de grilla y compara las predicciones de esas imágenes con el valor real
    del radio censal.

    Parameters:
    -----------
    df_test: pd.DataFrame, dataframe con los metadatos de las imágenes del conjunto de test
    tiles: int, cantidad de imágenes a generar por lado
    size: int, tamaño de la imagen a generar, en píxeles
    resizing_size: int, tamaño al que se redimensiona la imagen
    bias: int, cantidad de píxeles que se mueve el punto aleatorio de las tiles
    to8bit: bool, si es True, convierte la imagen a 8 bits

    Returns:
    --------
    mse: float, error cuadrático medio del conjunto de predicción

    """
    import random

    if verbose == False:
        blockPrint()

    # Inicializo arrays
    batch_images = np.empty((0, resizing_size, resizing_size, 4))
    batch_link_names = np.empty((0))
    batch_real_values = np.empty((0))
    batch_predictions = np.empty((0))
    all_link_names = np.empty((0))
    all_predictions = np.empty((0))
    all_real_values = np.empty((0))

    # Filtro Radios demasiado grandes (tardan horas en generar la cuadrícula y es puro campo...)
    if trim_size: 
        df_test = df_test[df_test["AREA"] <= 200000]  # Remove rc that are too big
    links = df_test["link"].unique()
    # links = random.sample(links, 30)
    len_links = len(links)

    # Creo la carpeta de test
    test_folder = rf"{path_satelites}/test_size{size}_tiles{tiles}_sample{sample}"
    os.makedirs(test_folder, exist_ok=True)
    print(len_links)

    # Loop por radio censal. Si está la imagen la usa, sino la genera.
    for n, link in enumerate(links):
        print(f"{link}: {n}/{len_links}")
        # Abre/genera la imagen
        file = rf"{test_folder}/test_{link}.npy"
        if os.path.isfile(file):
            images = np.load(file)
        else:
            print("Generando...")
            link_dataset = build_dataset.get_dataset_for_link(
                df_test, sat_img_datasets, link
            )
            print("listo")
            # if tiles == 1:
            images, bounds = build_dataset.get_gridded_images_for_link(
                link_dataset,
                df_test,
                link,
                tiles,
                size,
                resizing_size,
                sample,
                n_bands=n_bands,
                stacked_images=stacked_images
            )
            # else:  # If tiles >2 then the dataset is too big
            #     images, bounds = build_dataset.get_random_images_for_link(
            #         link_dataset,
            #         df_test,
            #         link,
            #         tiles,
            #         size,
            #         resizing_size,
            #         bias,
            #         sample,
            #         to8bit,
            #     )
            print("imagen generada")
            if len(images) == 0:
                # No images where returned from this census tract, so no error to compute...
                continue
            images = np.array(images)
            np.save(file, images)

        # Obtener el error de estimación del radio censal
        link_real_value = df_test.loc[df_test["link"] == link, "var"].values[0]

        # Agrega al batch de valores reales / imagenes para la prediccion
        q_images = images.shape[0]
        link_names = np.array([link] * q_images)
        real_values = np.array([link_real_value] * q_images)

        batch_images = np.concatenate([images, batch_images], axis=0)
        batch_link_names = np.concatenate([link_names, batch_link_names], axis=0)
        batch_real_values = np.concatenate([real_values, batch_real_values], axis=0)

        if batch_real_values.shape[0] > 128:
            batch_predictions = get_batch_predictions(model, batch_images)

            # Store data
            all_link_names = np.concatenate([all_link_names, batch_link_names])
            all_predictions = np.concatenate([all_predictions, batch_predictions])
            all_real_values = np.concatenate([all_real_values, batch_real_values])

            # Restore batches to empty
            batch_images = np.empty((0, resizing_size, resizing_size, 4))
            batch_link_names = np.empty((0))
            batch_real_values = np.empty((0))
            batch_predictions = np.empty((0))

    if verbose == False:
        enablePrint()

    # Creo dataframe para exportar:
    d = {
        "link": all_link_names,
        "predictions": all_predictions,
        "real_value": all_real_values,
    }
    df_preds = pd.DataFrame(data=d)

    df_preds["mean_prediction"] = df_preds.groupby(by="link").predictions.transform(
        "mean"
    )
    df_preds["error"] = df_preds["mean_prediction"] - df_preds["real_value"]
    df_preds["sq_error"] = df_preds["error"] ** 2
    mse = df_preds.drop_duplicates(subset=["link"]).sq_error.mean()

    return df_preds, mse


def compute_predictions_dataset(
    model,
    metadata,
    tiles,
    size,
    resizing_size,
    bias,
    sample,
    to8bit,
    verbose=False,
    trim_size=True,
):
    """
    Calcula predicciones y errores para un dataset determinado.

    Carga las bases necesarias, itera sobre los radios censales, genera las imágenes
    en forma de grilla y compara las predicciones de esas imágenes con el valor real
    del radio censal.

    Parameters:
    -----------
    metadata: pd.DataFrame, dataframe con los metadatos de las imágenes
    tiles: int, cantidad de imágenes a generar por lado
    size: int, tamaño de la imagen a generar, en píxeles
    resizing_size: int, tamaño al que se redimensiona la imagen
    bias: int, cantidad de píxeles que se mueve el punto aleatorio de las tiles
    to8bit: bool, si es True, convierte la imagen a 8 bits

    Returns:
    --------
    mse: float, error cuadrático medio del conjunto de predicción

    """
    import random

    if verbose == False:
        blockPrint()

    # Inicializo arrays
    batch_images = np.empty((0, resizing_size, resizing_size, 4))
    batch_link_names = np.empty((0))
    batch_real_values = np.empty((0))
    batch_predictions = np.empty((0))
    all_link_names = np.empty((0))
    all_predictions = np.empty((0))
    all_real_values = np.empty((0))

    # Cargar bases de datos
    datasets, extents = build_dataset.load_satellite_datasets()
    icpag = build_dataset.load_icpag_dataset()
    icpag = build_dataset.assign_links_to_datasets(icpag, extents, verbose=False)

    # Filtro Radios demasiado grandes (tardan horas en generar la cuadrícula y es puro campo...)
    if trim_size:
        icpag = icpag[icpag["AREA"] <= 4000000]  # Remove rc that are too big
    links = metadata["link"].astype(str).str.zfill(9).unique()
    links = [link for link in links if link in icpag.link.unique()]
    # links = random.sample(links, 30)
    len_links = len(links)

    # Creo la carpeta de test
    test_folder = rf"{path_dataout}/test_size{size}_tiles{tiles}_sample{sample}"
    os.makedirs(test_folder, exist_ok=True)

    # Loop por radio censal. Si está la imagen la usa, sino la genera.
    for n, link in enumerate(links):
        print(f"{link}: {n}/{len_links}")
        # Abre/genera la imagen
        file = rf"{test_folder}/test_{link}.npy"
        if os.path.isfile(file):
            images = np.load(file)
        else:
            link_dataset = build_dataset.get_dataset_for_link(icpag, datasets, link)
            images, points, bounds = build_dataset.get_gridded_images_for_link(
                link_dataset,
                icpag,
                link,
                tiles,
                size,
                resizing_size,
                bias,
                sample,
                to8bit,
            )
            if len(images) == 0:
                # No images where returned from this census tract, so no error to compute...
                continue
            images = np.array(images)
            np.save(file, images)

        # Obtener el error de estimación del radio censal
        link_real_value = metadata.loc[metadata["link"] == int(link), "var"].values[0]

        # Agrega al batch de valores reales / imagenes para la prediccion
        q_images = images.shape[0]
        link_names = np.array([link] * q_images)
        real_values = np.array([link_real_value] * q_images)

        batch_images = np.concatenate([images, batch_images], axis=0)
        batch_link_names = np.concatenate([link_names, batch_link_names], axis=0)
        batch_real_values = np.concatenate([real_values, batch_real_values], axis=0)

        if batch_real_values.shape[0] > 128:
            batch_predictions = get_batch_predictions(model, batch_images)

            # Store data
            all_link_names = np.concatenate([all_link_names, batch_link_names])
            all_predictions = np.concatenate([all_predictions, batch_predictions])
            all_real_values = np.concatenate([all_real_values, batch_real_values])

            # Restore batches to empty
            batch_images = np.empty((0, resizing_size, resizing_size, 4))
            batch_link_names = np.empty((0))
            batch_real_values = np.empty((0))
            batch_predictions = np.empty((0))

    if verbose == False:
        enablePrint()

    # Creo dataframe para exportar:
    d = {
        "link": all_link_names,
        "predictions": all_predictions,
        "real_value": all_real_values,
    }
    df_preds = pd.DataFrame(data=d)

    df_preds["mean_prediction"] = df_preds.groupby(by="link").predictions.transform(
        "mean"
    )
    df_preds["error"] = df_preds["mean_prediction"] - df_preds["real_value"]
    df_preds["sq_error"] = df_preds["error"] ** 2
    mse = df_preds.drop_duplicates(subset=["link"]).sq_error.mean()

    return df_preds, mse


def compute_custom_loss_all_epochs(
    models_dir,
    model_name,
    tiles,
    size,
    resizing_size,
    n_epochs=20,
    n_bands=4,
    stacked_images=[1],
    generate=False,
    verbose=False
):
    """
    Calcula el ECM del conjunto de predicción.

    Carga las bases necesarias, itera sobre los radios censales, genera las imágenes
    en forma de grilla y compara las predicciones de esas imágenes con el valor real
    del radio censal.

    Parameters:
    -----------
    tiles: int, cantidad de imágenes a generar por lado
    size: int, tamaño de la imagen a generar, en píxeles
    resizing_size: int, tamaño al que se redimensiona la imagen
    bias: int, cantidad de píxeles que se mueve el punto aleatorio de las tiles
    to8bit: bool, si es True, convierte la imagen a 8 bits

    Returns:
    --------
    mse: float, error cuadrático medio del conjunto de predicción

    """
    import geopandas as gpd
    from tqdm import tqdm
    savename = model_name
    mse_epochs = {epoch: None for epoch in range(n_epochs)}
    if verbose == False:
        blockPrint()

    # Load data
    print("Loading data...")
    sat_img_datasets, extents = build_dataset.load_satellite_datasets()
    df_test = gpd.read_feather(rf"{path_dataout}/test_datasets/test_dataframe.feather")
    print("Data loaded!")

    # dir of the images
    stacked_names = '-'.join(str(x) for x in stacked_images) # Transforms list [1,2] to string like "1-2"
    test_folder = rf"{path_dataout}/test_datasets/test_size{size}_tiles{tiles}_stacked{stacked_names}"

    # Genero las imágenes
    if generate:
        print("Generando imágenes en grilla...")
        test_folder = generate_gridded_images(
            df_test,
            sat_img_datasets,
            test_folder,
            tiles,
            size,
            resizing_size,
            n_bands,
            stacked_images
        )

    links = np.load(rf"{test_folder}/valid_links.npy")

    # Cargo todas las imágenes en memoria
    print('Cargando arrays en memoria...')
    # blockPrint()
    link_names = []
    real_values = []
    images = []
    for link in tqdm(links):
        # Obtener las imágenes del radio censal
        link_real_value = df_test.loc[df_test["link"] == link, "var"].values[0]
        link_images = np.load(rf"{test_folder}/test_{link}.npy")
        q_images = link_images.shape[0]
        
        link_names += [link] * q_images
        real_values += [link_real_value]*q_images
        images += [link_images]
        
    # Agrega al batch de valores reales / imagenes para la prediccion
    images = np.concatenate(images, axis=0)
    link_names = np.array(link_names)
    real_values = np.array(real_values)
    print("Arrays cargados!")
                
    for epoch in range(0, n_epochs):
        model = tf.keras.models.load_model(
            f"{models_dir}/{model_name}_{epoch}", compile=True
        )
        predictions = get_batch_predictions(model, images)

        # Creo dataframe para exportar:
        d = {
            "link": link_names,
            "predictions": predictions,
            "real_value": real_values,
        }
        df_preds = pd.DataFrame(data=d)

        df_preds["mean_prediction"] = df_preds.groupby(by="link").predictions.transform(
            "mean"
        )
        df_preds["error"] = df_preds["mean_prediction"] - df_preds["real_value"]
        df_preds["sq_error"] = df_preds["error"] ** 2
        mse = df_preds.drop_duplicates(subset=["link"]).sq_error.mean()  
        
        # enablePrint()
        print(f"Epoch {epoch}/{n_epochs}: True Mean Squared Error: {mse}")

        # Store MSE value in dict and full predictions
        mse_epochs[epoch] = mse
        df_preds.to_csv(
            f"{path_dataout}/models_by_epoch/{savename}/{savename}_{epoch}.csv"
        )

    pd.DataFrame().from_dict(mse_epochs, orient="index", columns=["MSE"])\
        .to_csv(f"{path_dataout}/models_by_epoch/{savename}/mse_over_epochs.csv")


if __name__ == "__main__":
    size = 128*3
    compute_custom_loss_all_epochs(
        models_dir=rf"/mnt/d/Maestría/Tesis/Repo/data/data_out/models_by_epoch/mobnet_v3_size{size}_tiles1_sample5",
        model_name=f"mobnet_v3_size{size}_tiles1_sample5",
        tiles=1,
        size=size,
        resizing_size=128,
        n_epochs=500,
        n_bands=4,
        stacked_images=[1],
        verbose=True,
        generate=True,
    )