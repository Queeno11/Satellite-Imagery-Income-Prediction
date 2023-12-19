##############      Configuración      ##############
import gc, sys

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


assert len(sys.argv) == 3, "Se esperan 2 argumentos: 1) Parte, 2) Total de partes"
parte = int(sys.argv[1])
total_partes = int(sys.argv[2])

env = dotenv_values("globals.env")

proyecto = 'Códigos'
subproyecto = 'Testing - Deep Learning con MapTilesDownloader'

globales = plantilla(proyecto=proyecto, 
                     subproyecto=subproyecto,
                     path_proyectos=env['PATH_PROYECTOS']
           )

path_proyecto   = globales[0]     # Ubicación de la carpeta del Proyecto
path_datain     = globales[1]
path_dataout    = globales[2]     # Bases procesadas por tus scripts
path_scripts    = globales[3]
path_figures    = globales[4]     # Output para las figuras/gráficos
path_maps       = globales[5]     # Output para los mapas (html o imagen)
path_tables     = globales[6]     # Output para las tablas (imagen o excel)
path_programas  = globales[7]


icpag = gpd.read_file(r"R:\Shapefiles\ICPAG\Sin barrios pop y cerr\Aglomerados de mas de 500k habitantes\base_icpag_500k.shp")
icpag = icpag.to_crs(epsg=3857)
icpag = icpag.reset_index(drop=True)
icpag_amba = icpag[icpag.AMBA_legal == 1]
icpag_amba = icpag_amba[icpag_amba.area <= 1_000_000] # Aprox p90
icpag_amba = icpag_amba.reset_index(drop=True)

map_provider = cimgt.GoogleTiles(url='https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}')

icpag_amba.geometry = icpag_amba.centroid.buffer(100)


def random_point_from_geometry(gdf_shape):
    """Generates a random point within the bounds of a GeoDataFrame."""

    gdf_obs = gdf_shape.copy()

    # Get bounds of the shapefile's polygon
    bbox = gdf_obs.bounds
   
    while 0==0:
        # generate random data within the bounds
        x = np.random.uniform(bbox['minx'], bbox['maxx'], 1)
        y = np.random.uniform(bbox['miny'], bbox['maxy'], 1)

        # convert them to a points GeoSeries
        gdf_points = gpd.GeoSeries(gpd.points_from_xy(x, y))
        # only keep those points within polygons
        gdf_points = gdf_points[gdf_points.within(gdf_obs.unary_union)]
        if len(gdf_points) > 0:
            break
            
    polygon = gdf_points.buffer(100)
    polygon = polygon.set_crs(epsg=3857)

    gdf_obs.loc[:,'geometry'] = polygon 
    return gdf_obs

def create_map_from_geometry(
    icpag, index, 
    map_provider=map_provider, zoom=18, map_name='ESRI_WI',  
    path_output=path_datain,
    sample_size=1):

    Image.MAX_IMAGE_PIXELS = 2160000 
    
    for i in range(0,sample_size):
        try:
            # Reduzco el polygono para que sea aprox una manzana
            # polygon = random_point_from_geometry(icpag.iloc[index:index+1,:])
            polygon = icpag.iloc[index:index+1,:]

            # Genero la máscara para el gráfico y obtengo el extent
            link = polygon.at[index,'link']
            bbox = polygon.bounds
            geom =box(*bbox.values[0])
            mask = polygon.copy()
            mask['geometry'] = geom

            # Gráfico
            # The pylab figure manager will be bypassed in this instance.
            # This means that `fig` will be garbage collected as you'd expect.
            fig = Figure(dpi=300, figsize=(10,10), linewidth=0)
            ax = fig.add_axes(
                    [0,0,1,1], 
                    projection=crs.epsg(3857),  facecolor = 'black'
                )

            # Limita la visualización a un área determinada
            ax.set_extent(
                [bbox['minx'], bbox['maxx'], bbox['miny'], bbox['maxy']],
                crs=crs.epsg(3857)
            )

            # Agrego mapa de fondo
            ax.add_image(map_provider, zoom)

            # Quito bordes y grilla
            ax.set(frame_on=False)

            # # Añade la máscara
            # mask.difference(polygon).plot(ax=ax, facecolor='black', edgecolor='black', linewidth=0.0) 
            # fig.add_axes(ax)
            
            fig.savefig(fr"{path_output}\Imagenes descargadas\{map_name}_{link}_{index}_{i}.tiff")
            del fig
            gc.collect()
            
        except Exception as e:
            print(e)
            continue

import pandas as pd
# icpag_amba_a = icpag_amba.set_index('link').loc['020090206':'064271706',:]
# icpag_amba_b = icpag_amba.set_index('link').loc['064412304':,:]
# icpag_amba = pd.concat(
#     [icpag_amba_a,icpag_amba_b]).reset_index()
        
list = np.array_split(range(icpag_amba.shape[0]), total_partes)

from tqdm import tqdm
for index in tqdm(list[parte]):
    create_map_from_geometry(icpag_amba, index)