##############      BBOX a graficar    ##############    
# Test area 2
VILLA_31 = [[-58.3942356256,-34.5947320752],[-58.3679285222,-34.5947320752],[-58.3679285222,-34.5760771024],[-58.3942356256,-34.5760771024],[-58.3942356256,-34.5947320752]]
VILLA_21_24 = [[-58.4108331428,-34.6551111569],[-58.3944394813,-34.6551111569],[-58.3944394813,-34.6437196838],[-58.4108331428,-34.6437196838],[-58.4108331428,-34.6551111569]]
ADROGUE_CHICO = [[-58.409929469,-34.817989341],[-58.3966363097,-34.817989341],[-58.3966363097,-34.8070532351],[-58.409929469,-34.8070532351],[-58.409929469,-34.817989341]]
# Test area 1
PARQUE_LELOIR = [[-58.67620171,-34.6352316984],[-58.6650723329,-34.6352316984],[-58.6650723329,-34.6252500848],[-58.67620171,-34.6252500848],[-58.67620171,-34.6352316984]]
BELLAVISTA = [[[-58.7013553135,-34.5908167952],[-58.6871646388,-34.5908167952],[-58.6871646388,-34.5802173929],[-58.7013553135,-34.5802173929],[-58.7013553135,-34.5908167952]]]
NORDELTA = [[-58.6856032553,-34.4167915279],[-58.6720706202,-34.4167915279],[-58.6720706202,-34.4065712602],[-58.6856032553,-34.4065712602],[-58.6856032553,-34.4167915279]]
NORDELTA2 = [[-58.6649109046,-34.4174228815],[-58.6445404177,-34.4174228815],[-58.6445404177,-34.4011242703],[-58.6649109046,-34.4011242703],[-58.6649109046,-34.4174228815]]

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
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import earthpy.plot as ep
from shapely import Point, Polygon


def to_square(polygon):
    from math import sqrt
    
    minx, miny, maxx, maxy = polygon.bounds
    
    # get the centroid
    centroid = [(maxx+minx)/2, (maxy+miny)/2]
    # get the diagonal
    diagonal = sqrt((maxx-minx)**2+(maxy-miny)**2)
    
    return Point(centroid).buffer(diagonal/sqrt(2.)/2., cap_style=3)


def crop_dataset_by_polygon(ds, poly):
    
    x, y = poly.exterior.xy
    x_min = np.array(x).min()
    x_max = np.array(x).max()
    y_min = np.array(y).min()
    y_max = np.array(y).max()

    area_selected = ds.sel(x=slice(x_min, x_max), y=slice(y_max, y_min)).band_data.values

    return area_selected

def ds_plot_example(ds, poly, ax):
    band_data_image = crop_dataset_by_polygon(ds, poly)
    ep.plot_rgb(band_data_image, ax=ax) 

def gdf_plot_example(gdf, var, poly, ax, vmin=None, vmax=None):
    gdf.plot(column=var, cmap="Spectral", vmin=vmin, vmax=vmax, ax=ax, aspect="equal")

    x, y = poly.exterior.xy
    x_min = np.array(x).min()
    x_max = np.array(x).max()
    y_min = np.array(y).min()
    y_max = np.array(y).max()

    ax.set_axis_off()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    
def plot_example(bbox, modelname, img_savename):

    ds = xr.open_dataset(f"{path_satelites}/Compressed/2013/pansharpened_6741387101_R1C1.tif") # FIXME: elegir automaticamente el dataset
    prediction = gpd.read_parquet(f"{path_dataout}/gridded_predictions/{modelname}/pansharpened_6741387101_R1C1.parquet") # FIXME: aca seguro tengo que cambiar esto
    icpag = gpd.read_feather(f"{path_datain}/census_tracts_with_indicators.feather")

    poly = to_square(Polygon(bbox))

    import earthpy.plot as ep
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    ds_plot_example(ds, poly, ax=axs[0]) 
    axs[0].set_title("Imagen satelital")
    gdf_plot_example(prediction, var="predictions", poly=poly, ax=axs[1])
    axs[1].set_title("Ingreso predicho por imagenes satelitales")
    gdf_plot_example(icpag, var="ln_pred_inc_mean", poly=poly, ax=axs[2])
    axs[2].set_title("Ingreso estructural por small area")

    fig.tight_layout()
    # plt.savefig(savename)
    
    
if __name__ == "__main__":

    plot_example(VILLA_31, "")