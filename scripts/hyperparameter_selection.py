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

import run_model
import custom_models
import build_dataset
import grid_predictions
import true_metrics
import utils

import os
import sys
import scipy
import random
import pandas as pd
import xarray as xr
import tensorflow as tf

try:
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print("No GPU set. Is the GPU already initialized?")


def get_true_val_loss(
    model_name,
    sat_data,
    tiles,
    size,
    resizing_size,
    n_epochs,
    n_bands,
    stacked_images,
    generate,
):

    savename = run_model.generate_savename(
        model_name, image_size, tiles, sample_size, extra, stacked_images
    )

    all_years_datasets, all_years_extents, df = run_model.open_datasets(
        sat_data=sat_data, years=[2013]
    )

    metrics_epochs = true_metrics.compute_custom_loss_all_epochs(
        rf"{path_dataout}/models_by_epoch/{savename}",
        savename,
        all_years_datasets[2013],
        tiles,
        size,
        resizing_size,
        "val",
        n_epochs,
        n_bands,
        stacked_images,
        generate,
        verbose=True,
    )

    return metrics_epochs


if __name__ == "__main__":
    import warnings

    image_size = 256  # FIXME: Creo que solo anda con numeros pares, alguna vez estaría bueno arreglarlo...
    resizing_size = 128
    sample_size = 5
    tiles = 1
    stacked_images = [1]
    epochs = 200

    variable = "ln_pred_inc_mean"
    kind = "reg"
    model = "effnet_v2M"
    path_repo = r"/mnt/d/Maestría/Tesis/Repo/"
    extra = ""
    sat_data = "pleiades"

    if sat_data == "pleiades":
        years = [2013, 2018, 2022]
        nbands = 4
    elif sat_data == "landsat":
        years = [2013]
        nbands = 10
        image_data = 32
        resizing_size = 32

    get_true_val_loss(
        model,
        sat_data,
        tiles,
        image_size,
        resizing_size,
        epochs,
        nbands,
        stacked_images,
        generate=True,
    )
