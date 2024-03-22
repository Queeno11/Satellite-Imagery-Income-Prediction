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
import matplotlib.pyplot as plt

try:
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print("No GPU set. Is the GPU already initialized?")


def get_true_val_loss(params):

    params = run_model.fill_params_defaults(params)

    model_name = params["model_name"]
    kind = params["kind"]
    weights = params["weights"]
    image_size = params["image_size"]
    resizing_size = params["resizing_size"]
    tiles = params["tiles"]
    nbands = params["nbands"]
    stacked_images = params["stacked_images"]
    sample_size = params["sample_size"]
    small_sample = params["small_sample"]
    n_epochs = params["n_epochs"]
    learning_rate = params["learning_rate"]
    sat_data = params["sat_data"]
    years = params["years"]
    extra = params["extra"]

    savename = run_model.generate_savename(
        model_name, image_size, learning_rate, stacked_images, years, extra
    )
    metrics_path = rf"{path_dataout}/models_by_epoch/{savename}/{savename}_test_metrics_over_epochs.csv"
    if not os.path.exists(metrics_path):

        all_years_datasets, all_years_extents, df = run_model.open_datasets(
            sat_data=sat_data, years=[2013]
        )

        metrics_epochs = true_metrics.compute_custom_loss_all_epochs(
            rf"{path_dataout}/models_by_epoch/{savename}",
            savename,
            all_years_datasets[
                2013
            ],  # Only 2013 because I want to test with the ground truth images...
            tiles,
            image_size,
            resizing_size,
            "test",
            n_epochs,
            nbands,
            stacked_images,
            generate=False,
            verbose=True,
        )

    df = pd.read_csv(
        metrics_path,
        index_col="epoch",
        usecols=["epoch", "mse_train", "mse_test_rc"],
        nrows=100,
    )

    return df


def compute_experiment_results(options):
    """Compute true loss if needed and plot comparison between the different options"""
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    data_for_plot = {}

    for name, params in options.items():

        df = get_true_val_loss(params)

        sns.lineplot(df["mse_train"], ax=ax[0], label=f"Baseline ({name})")
        sns.lineplot(df["mse_test_rc"], ax=ax[1], label=f"Baseline ({name})")
        data_for_plot[name] = df

    ax[0].set_ylim(0, 0.3)
    ax[0].set_ylabel("")
    ax[0].set_title("ECM Entrenamiento")

    ax[1].set_ylim(0, 0.3)
    ax[1].set_ylabel("")
    ax[1].set_title("ECM Conjunto de Prueba (media radio censal)")

    plt.savefig(rf"{path_outputs}/{experiment_name}.png", dpi=300, bbox_inches="tight")
    print("Se creó la imagen " + rf"{path_outputs}/{experiment_name}.png")

if __name__ == "__main__":
    import warnings

    experiment_name = "learning_rates"
    options = {
        "lr: 0.001": {"learning_rate": 0.001},
        "lr: 0.0001": {"learning_rate": 0.0001},
        "lr: 0.00001": {"learning_rate": 0.00001},
    }

    compute_experiment_results(options)
