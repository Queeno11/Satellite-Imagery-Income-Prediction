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
import keras
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import earthpy.plot as ep
from shapely import Point, Polygon
import build_dataset
import true_metrics
import utils

def get_areas_for_evaluation():
    a_graficar = {
        # "AMBA" : [[-58.6715814736,-34.7982854506],[-58.2193109518,-34.7982854506],[-58.2193109518,-34.4570175785],[-58.6715814736,-34.4570175785],[-58.6715814736,-34.7982854506]],
        # Test area 2
        "VILLA_31" : [[-58.3942356256,-34.5947320752],[-58.3679285222,-34.5947320752],[-58.3679285222,-34.5760771024],[-58.3942356256,-34.5760771024],[-58.3942356256,-34.5947320752]],
        "VILLA_21_24" : [[-58.4108331428,-34.6551111569],[-58.3944394813,-34.6551111569],[-58.3944394813,-34.6437196838],[-58.4108331428,-34.6437196838],[-58.4108331428,-34.6551111569]],
        "ADROGUE_CHICO" : [[-58.409929469,-34.817989341],[-58.3966363097,-34.817989341],[-58.3966363097,-34.8070532351],[-58.409929469,-34.8070532351],[-58.409929469,-34.817989341]],
        # Test area 1
        "PARQUE_LELOIR" : [[-58.67620171,-34.6352316984],[-58.6650723329,-34.6352316984],[-58.6650723329,-34.6252500848],[-58.67620171,-34.6252500848],[-58.67620171,-34.6352316984]],
        "BELLAVISTA" : [[-58.7013553135,-34.5908167952],[-58.6871646388,-34.5908167952],[-58.6871646388,-34.5802173929],[-58.7013553135,-34.5802173929],[-58.7013553135,-34.5908167952]],
        "NORDELTA" : [[-58.6856032553,-34.4167915279],[-58.6720706202,-34.4167915279],[-58.6720706202,-34.4065712602],[-58.6856032553,-34.4065712602],[-58.6856032553,-34.4167915279]],
        "NORDELTA2" : [[-58.6649109046,-34.4174228815],[-58.6445404177,-34.4174228815],[-58.6445404177,-34.4011242703],[-58.6649109046,-34.4011242703],[-58.6649109046,-34.4174228815]],
    }
    # Fix issue with Adrogue chico: doesnt fit fully on any dataset
    a_graficar["ADROGUE_CHICO"] = [[x, y+.004] for x,y in a_graficar["ADROGUE_CHICO"]]
    
    return a_graficar

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
    gdf["plot_var"] = pd.qcut(gdf[var], 10) 
    gdf.plot(column="plot_var", cmap="Spectral", ax=ax, aspect="equal")

    x, y = poly.exterior.xy
    x_min = np.array(x).min()
    x_max = np.array(x).max()
    y_min = np.array(y).min()
    y_max = np.array(y).max()

    ax.set_axis_off()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
def generate_grid(savename, image_size, resizing_size, nbands):
    # Cargo datasets
    hist_df = pd.read_csv(fr"{path_dataout}/models_by_epoch/{savename}/{savename}_metrics_over_epochs.csv")
    best_epoch = hist_df[hist_df.mse_test_rc.min()==hist_df.mse_test_rc].index.item()
    datasets, extents = build_dataset.load_satellite_datasets()
    icpag = build_dataset.load_icpag_dataset()
    
    # Cargo modelo
    model_path = f"{path_dataout}/models_by_epoch/{savename}/{savename}_{best_epoch}"
    model = keras.models.load_model(model_path)  # load the model from file

    # Genero la grilla
    grid_preds = true_metrics.get_gridded_predictions_for_grid(
        model, datasets, extents, icpag, image_size, resizing_size, n_bands=nbands
    )
    
    # Guardo la grilla
    grid_preds_folder = rf"{path_dataout}/gridded_predictions/{savename}"
    os.makedirs(grid_preds_folder, exist_ok=True)
    grid_preds.to_parquet(rf"{grid_preds_folder}/{savename}_{best_epoch}_predictions.parquet")
    
    return grid_preds, datasets, extents

def plot_example(grid_preds, bbox, modelname, datasets, extents, img_savename):
    
    # BBox a poligono
    poly = to_square(Polygon(bbox))
    
    # Carga datasets a memoria
    name = utils.get_dataset_for_polygon(poly, extents)

    ds = datasets[name] 
    icpag = build_dataset.load_icpag_dataset(variable="ln_pred_inc_mean")

    import earthpy.plot as ep
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    ds_plot_example(ds, poly, ax=axs[0]) 
    axs[0].set_title("Imagen satelital")
    gdf_plot_example(grid_preds, var="prediction", poly=poly, ax=axs[1])#, vmin=icpag["ln_pred_inc_mean"].quantile(.1), vmax=icpag["ln_pred_inc_mean"].quantile(.9))
    axs[1].set_title("Ingreso predicho por imagenes satelitales")
    gdf_plot_example(icpag, var="var", poly=poly, ax=axs[2])#, vmin=icpag["ln_pred_inc_mean"].quantile(.1), vmax=icpag["ln_pred_inc_mean"].quantile(.9))
    axs[2].set_title("Ingreso estructural por small area")
    fig.tight_layout()
    savepath = f"{path_outputs}/{modelname}"
    os.makedirs(savepath, exist_ok=True)
    plt.savefig(f"{path_outputs}/{modelname}/{modelname}_{img_savename}", bbox_inches='tight', dpi=300)
    print("Se creó la imagen: ", f"{path_outputs}/{modelname}/{modelname}_{img_savename}.png")
    
if __name__ == "__main__":
    
    image_size = 128*2 # FIXME: VER SI ESTO FUNCIONA!!
    sample_size = 1
    resizing_size = 128
    tiles = 1
    n_bands = 4

    kind = "reg"
    model_name= "mobnet_v3"
    path_repo = r"/mnt/d/Maestría/Tesis/Repo/"
    extra = "_nostack"

    best_epoch = 99
    
    model_savename = f"{model_name}_size{image_size}_tiles{tiles}_sample{sample_size}{extra}"
    
    # ## Cargo datasets
    datasets, extents = build_dataset.load_satellite_datasets()
    icpag = build_dataset.load_icpag_dataset()

    ## Carga modelo
    model_path = f"{path_dataout}/models_by_epoch/{model_savename}/{model_savename}_{best_epoch}"
    model = keras.models.load_model(model_path)  # load the model from file
    
    ## Armar grilla de predicciones:
    grid_preds = true_metrics.get_gridded_predictions_for_grid(
        model, datasets, extents, icpag, image_size, resizing_size, n_bands=n_bands
    )
    grid_preds_folder = rf"{path_dataout}/gridded_predictions/{model_savename}"
    os.makedirs(grid_preds_folder, exist_ok=True)
    grid_preds.to_parquet(rf"{grid_preds_folder}/{model_savename}_{best_epoch}_predictions.parquet")
    
    ##############      BBOX a graficar    ##############    
    a_graficar = get_areas_for_evaluation()
    for zona, bbox in a_graficar.items():
        plot_example(bbox, model_savename, datasets, extents, best_epoch, zona)
