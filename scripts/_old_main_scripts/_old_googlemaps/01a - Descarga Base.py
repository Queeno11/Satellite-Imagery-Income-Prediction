#### HOW TO RUN THIS SCRIPT ####
# 1) Open CMD
# 2) Run the following command:
#    python "R:\Tesis Nico\Códigos\scripts\Testing - Deep Learning con MapTilesDownloader\01a - Descarga Base.py" <map_provider> <total_partes> <parte>
#
#    <map_provider> = ESRI, GMaps o both
#    <total_partes> = int (1, 2, 3, ...)
#    <parte> = int (1, 2, 3, ...)
#
#    Ejemplo: python "R:\Tesis Nico\Códigos\scripts\Testing - Deep Learning con MapTilesDownloader\01a - Descarga Base.py" ESRI 4 1
#
# The generated images will be saved with the following name:
#    <map_provider>_<link>_<index>_<sample>.png
#
#    <map_provider> = ESRI, GMaps o both
#    <link> = str (ej: 064415285, siempre con 9 dígitos). ID del radio censal.
#    <index> = int (ej: 0, 1, 2, ...). Indice del radio censal en el scrapeo. Sirve para retomar la descarga si se corta el proceso.
#    <sample> = int (ej: 1, 2, ...). Indice de la muestra. Número de imagen descargada del radio censal.

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
), "Se esperan 3 argumentos: 1) Map Provider: ESRI, GMaps o both, 2) Total de partes, 3) Parte"

selection = str(sys.argv[1])
total_partes = int(sys.argv[2])
parte = int(sys.argv[3]) - 1  # -1 porque el primer elemento es 0

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


icpag = gpd.read_feather(f"{path_dataout}\census_tracts_with_indicators.feather")
icpag = icpag.to_crs(epsg=3857)
# icpag = icpag[icpag.area <= 1_000_000]  # Aprox p90
# icpag_amba = icpag[icpag.AMBA_legal == 1]
icpag = icpag.reset_index(drop=True)

if selection == "both":
    map_providers = {
        "ESRI": cimgt.GoogleTiles(
            url="https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        "GMaps": cimgt.GoogleTiles(style="satellite"),
    }
elif selection == "ESRI":
    map_providers = {
        "ESRI": cimgt.GoogleTiles(
            url="https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
    }
elif selection == "GMaps":
    map_providers = {
        "GMaps": cimgt.GoogleTiles(style="satellite"),
    }
else:
    raise ValueError("map_provider debe ser ESRI o GMaps")


def random_point_from_geometry(gdf_slice, size=100):
    """Generates a random point within the bounds of a GeoDataFrame."""

    gdf_slice = gdf_slice.copy()

    # Get bounds of the shapefile's polygon
    bbox = gdf_slice.bounds

    while 0 == 0:
        # generate random data within the bounds
        x = np.random.uniform(bbox["minx"], bbox["maxx"], 1)
        y = np.random.uniform(bbox["miny"], bbox["maxy"], 1)

        # convert them to a points GeoSeries
        gdf_points = gpd.GeoSeries(gpd.points_from_xy(x, y))
        # only keep those points within polygons
        gdf_points = gdf_points[gdf_points.within(gdf_slice.unary_union)]
        if len(gdf_points) > 0:
            # If one point is found, stop the loop
            break

    gdf_slice = gdf_slice.drop(columns="geometry")
    gdf_slice = gpd.GeoDataFrame(gdf_slice, geometry=gpd.points_from_xy(x, y))
    gdf_slice = gdf_slice.set_crs(epsg=3857)
    gdf_slice.geometry = gdf_slice.centroid.buffer(size)

    return gdf_slice


def create_map_from_geometry(
    icpag,
    index,
    folder,
    map_provider,
    map_name,
    size=100,
    path_output=path_datain,
    sample_size=1,
    my_dpi=96,
):

    """Exporto imgs de 512x512.

    Matplotlib doesn't work with pixels directly, but rather physical sizes and DPI.
    If you want to display a figure with a certain pixel size, you need to know the DPI of your monitor.
    For example this link (https://www.infobyip.com/detectmonitordpi.php) will detect that for you.
    """

    from pathlib import Path

    # create folder if it doesn't exist
    Path(f"{path_datain}\{folder}").mkdir(parents=True, exist_ok=True)

    for i in range(1, sample_size + 1):
        try:
            # Reduzco el polygono para que sea aprox una manzana
            polygon = random_point_from_geometry(icpag.iloc[index : index + 1, :], size)
            # polygon = icpag.iloc[index : index + 1, :]

            # Genero la máscara para el gráfico y obtengo el extent
            link = polygon.at[index, "link"]
            bbox = polygon.bounds
            geom = box(*bbox.values[0])
            mask = polygon.copy()
            mask["geometry"] = geom

            # Gráfico
            # The pylab figure manager will be bypassed in this instance.
            # This means that `fig` will be garbage collected as you'd expect.
            fig = Figure(dpi=my_dpi, figsize=(512 / my_dpi, 512 / my_dpi), linewidth=0)
            ax = fig.add_axes(
                [0, 0, 1, 1], projection=crs.epsg(3857), facecolor="black"
            )

            # Limita la visualización a un área determinada
            ax.set_extent(
                [bbox["minx"], bbox["maxx"], bbox["miny"], bbox["maxy"]],
                crs=crs.epsg(3857),
            )

            # Agrego mapa de fondo
            ax.add_image(map_provider, 19)

            # Quito bordes y grilla
            ax.set(frame_on=False)

            # # Añade la máscara
            # mask.difference(polygon).plot(ax=ax, facecolor='black', edgecolor='black', linewidth=0.0)
            # fig.add_axes(ax)
            is_urban = icpag.urban.iloc[index : index + 1].values[0] == 1
            if is_urban:
                fig.savefig(
                    rf"{path_datain}\{folder}\urban\{map_name}_{link}_{index}_{i}.jpg",
                    dpi=my_dpi,
                )
                del fig
                gc.collect()
            else:
                fig.savefig(
                    rf"{path_datain}\{folder}\rural\{map_name}_{link}_{index}_{i}.jpg",
                    dpi=my_dpi,
                )
                del fig
                gc.collect()
        except Exception as e:
            print(e)
            continue


import pandas as pd

# icpag_amba_a = icpag_amba.set_index('link').loc['020090206':'0642 71706',:]
# icpag_amba_b = icpag_amba.set_index('link').loc['064412304':,:]
# icpag_amba = pd.concat(
#     [icpag_amba_a,icpag_amba_b]).reset_index()

list = np.array_split(range(icpag.shape[0]), total_partes)

from tqdm import tqdm

for index in tqdm(list[parte]):

    ##### Single picture

    for map_name, map_provider in map_providers.items():
        # size radios 25, aprox 50x50m
        create_map_from_geometry(
            icpag,
            index,
            map_provider=map_provider,
            folder="g1_s25_n5",
            map_name=map_name,
            sample_size=5,
            size=25,
        )

    # for map_provider in map_providers:
    #     # size radios 50, aprox 100x100m
    #     create_map_from_geometry(
    #         icpag,
    #         index,
    #         map_provider=map_provider,
    #         folder="g1_s50_n5",
    #         map_name=map_name,
    #         sample_size=5,
    #         size=50,
    #     )

    for map_name, map_provider in map_providers.items():
        # size radios 75, aprox 150x150m
        create_map_from_geometry(
            icpag,
            index,
            map_provider=map_provider,
            folder="g1_s75_n5",
            map_name=map_name,
            sample_size=5,
            size=75,
        )

    for map_name, map_provider in map_providers.items():
        # size radios 125, aprox 250x250m
        create_map_from_geometry(
            icpag,
            index,
            map_provider=map_provider,
            folder="g1_s125_n5",
            map_name=map_name,
            sample_size=5,
            size=125,
        )
