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


