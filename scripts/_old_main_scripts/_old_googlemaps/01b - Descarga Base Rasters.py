#### CORRER EN CMD
# python "R:\Tesis Nico\Códigos\scripts\Testing - Deep Learning con MapTilesDownloader\01b - Descarga Base Rasters.py" ESRI 16 0
# con 16 partes y la parte 0, para el mapa ESRI
##############      Configuración      ##############
import gc, sys, os

from dotenv import dotenv_values
from plantilla import plantilla
import numpy as np
import pandas as pd
import geopandas as gpd

import cartopy.crs as crs
import cartopy.io.img_tiles as cimgt
from matplotlib.figure import Figure
from shapely.geometry import box
from PIL import Image


assert (
    len(sys.argv) == 4
), "Se esperan 3 argumentos: 1) Total de partes, 2) Parte,  3) Map Provider: ESRI o GMaps"
map_name = str(sys.argv[1])
total_partes = int(sys.argv[2])
parte = int(sys.argv[3])

env = dotenv_values(
    r"R:\Tesis Nico\Códigos\scripts\Testing - Deep Learning con MapTilesDownloader\globals.env"
)

proyecto = "Códigos"
subproyecto = "Testing - Deep Learning con MapTilesDownloader"

globales = plantilla(
    proyecto=proyecto, subproyecto=subproyecto, path_proyectos=env["PATH_PROYECTOS"]
)

path_proyecto = globales[0]  # Ubicación de la carpeta del Proyecto
path_datain = globales[1]
path_dataout = globales[2]  # Bases procesadas por tus scripts
path_scripts = globales[3]
path_figures = globales[4]  # Output para las figuras/gráficos
path_maps = globales[5]  # Output para los mapas (html o imagen)
path_tables = globales[6]  # Output para las tablas (imagen o excel)
path_programas = globales[7]

icpag = gpd.read_file(
    r"R:\Tesis Nico\Códigos\data\data_in\Rasters Tigre ejemplo rasterización.shp"
)
# icpag = gpd.read_feather(
#     r"R:\Tesis Nico\Códigos\data\data_in\Raster_para_predicción.feather"
# )
print("Base cargada")

icpag = icpag.to_crs(epsg=3857)
# icpag = icpag[icpag.area <= 1_000_000]  # Aprox p90
# icpag_amba = icpag[icpag.AMBA_legal == 1]
icpag = icpag.reset_index(drop=True)

if map_name == "ESRI":
    map_provider = cimgt.GoogleTiles(
        url="https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    )
elif map_name == "GMaps":
    map_provider = cimgt.GoogleTiles(style="satellite")
else:
    raise ValueError("map_provider debe ser ESRI o GMaps")

icpag.geometry = icpag.centroid.buffer(100)


def create_map_from_geometry(
    icpag,
    index,
    map_provider=map_provider,
    zoom=18,
    map_name="ESRI_WI",
    my_dpi=96,
):

    """Exporto imgs de 512x512.

    Matplotlib doesn't work with pixels directly, but rather physical sizes and DPI.
    If you want to display a figure with a certain pixel size, you need to know the DPI of your monitor.
    For example this link (https://www.infobyip.com/detectmonitordpi.php) will detect that for you.
    """
    # Reduzco el polygono para que sea aprox una manzana
    # polygon = random_point_from_geometry(icpag.iloc[index : index + 1, :])
    polygon = icpag.iloc[index : index + 1, :]

    # Genero la máscara para el gráfico y obtengo el extent
    id = polygon.at[index, "id"]
    bbox = polygon.bounds
    geom = box(*bbox.values[0])
    mask = polygon.copy()
    mask["geometry"] = geom

    # Gráfico
    # The pylab figure manager will be bypassed in this instance.
    # This means that `fig` will be garbage collected as you'd expect.
    fig = Figure(dpi=my_dpi, figsize=(512 / my_dpi, 512 / my_dpi), linewidth=0)
    ax = fig.add_axes([0, 0, 1, 1], projection=crs.epsg(3857), facecolor="black")

    # Limita la visualización a un área determinada
    ax.set_extent(
        [bbox["minx"], bbox["maxx"], bbox["miny"], bbox["maxy"]],
        crs=crs.epsg(3857),
    )

    # Agrego mapa de fondo
    ax.add_image(map_provider, zoom)

    # Quito bordes y grilla
    ax.set(frame_on=False)

    # # Añade la máscara
    # mask.difference(polygon).plot(ax=ax, facecolor='black', edgecolor='black', linewidth=0.0)
    # fig.add_axes(ax)

    fig.savefig(
        rf"{path_dataout}\inference\rasters_map_tigre\{map_name}_{id}.tiff",
        dpi=my_dpi,
    )
    del fig
    gc.collect()


import pandas as pd

# icpag_amba_a = icpag_amba.set_index('id').loc['020090206':'064271706',:]
# icpag_amba_b = icpag_amba.set_index('id').loc['064412304':,:]
# icpag_amba = pd.concat(
#     [icpag_amba_a,icpag_amba_b]).reset_index()

list = np.array_split(range(icpag.shape[0]), total_partes)

from tqdm import tqdm

for index in tqdm(list[parte]):
    create_map_from_geometry(icpag, index, map_name=map_name)
