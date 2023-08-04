import shapely
import numpy as np
import geopandas as gpd


def get_dataset_extent(ds):
    """Return a polygon with the extent of the dataset

    Params:
    ds: xarray dataset

    Returns:
    polygon: shapely polygon with the extent of the dataset
    """

    x_min = ds.x.min()
    x_max = ds.x.max()
    y_min = ds.y.min()
    y_max = ds.y.max()

    bbox = (x_min, y_min, x_max, y_max)

    # Turn bbox into a shapely polygon
    polygon = shapely.geometry.box(*bbox)

    return polygon


def random_point_from_geometry(gdf_slice, size=100):
    """Generates a random point within the bounds of a GeoDataFrame."""

    # Get bounds of the shapefile's polygon
    bbox = gdf_slice.bounds

    while 0 == 0:
        # generate random data within the bounds
        x = np.random.uniform(bbox["minx"], bbox["maxx"], 1)
        y = np.random.uniform(bbox["miny"], bbox["maxy"], 1)

        # convert them to a points GeoSeries
        gdf_points = gpd.GeoSeries(gpd.points_from_xy(x, y), crs=3857)
        # only keep those points within polygons
        gdf_points = gdf_points[gdf_points.within(gdf_slice.unary_union)]
        if len(gdf_points) > 0:
            # If one point is found, stop the loop
            return (x, y)


def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def find_nearest_raster(x_array, y_array, x_value, y_value, max_bias=0):
    """If bias is added, then the point is moved randomly within a circle of radius max_bias (in pixels)."""

    # Calculate the bias for each axis
    angle_bias = np.random.uniform(0, 360)
    # actual_bias = np.random.uniform(0, max_bias)
    actual_bias = max_bias
    x_bias = round(np.cos(angle_bias) * actual_bias)
    y_bias = round(np.sin(angle_bias) * actual_bias)

    # Get the nearest index for each axis and add the bias
    x_idx = find_nearest_idx(x_array, x_value) + x_bias
    y_idx = find_nearest_idx(y_array, y_value) + y_bias
    return x_idx, y_idx


def random_image_from_census_tract(ds, icpag, link, tiles=1, size=100, bias=4, to8bit=True):
    """Genera una imagen aleatoria de tamaño size centrada en un punto aleatorio del radio censal {link}.

    Parameters:
    -----------
    ds: xarray.Dataset, dataset con las imágenes de satélite
    icpag: geopandas.GeoDataFrame, shapefile con los radios censales
    link: str, 9 dígitos que identifican el radio censal
    size: int, tamaño de la imagen a generar, en píxeles
    tiles: int, cantidad de imágenes a generar por lado
    to8bit: bool, si es True, convierte la imagen a 8 bits

    Returns:
    --------
    image: numpy.ndarray, imagen de tamaño size x size
    point: tuple, coordenadas del punto seleccionado

    """

    images = []
    boundaries = []
    tile_size = size // tiles
    for tile in range(0, tiles**2):
        if tile == 0:
            max_bias = 0
        else:
            max_bias = bias*size

        # Obtengo un punto aleatorio del radio censal con un buffer de tamaño size
        x, y = random_point_from_geometry(icpag.loc[icpag["link"] == link], tile_size)
        point = (x[0], y[0])

        # Identifico el raster más cercano a este punto -- va a ser el centro de la imagen
        idx_x, idx_y = find_nearest_raster(ds.x, ds.y, x, y, max_bias=max_bias)

        # # Genero el cuadrado que quiero capturar en la imagen
        idx_x_min = round(idx_x - tile_size / 2)
        idx_x_max = round(idx_x + tile_size / 2)
        idx_y_min = round(idx_y - tile_size / 2)
        idx_y_max = round(idx_y + tile_size / 2)

        # Filtro el dataset para quedarme con esa imagen
        my_ds = ds.isel(x=slice(idx_x_min, idx_x_max), y=slice(idx_y_min, idx_y_max))

        image = my_ds.band_data.to_numpy().astype(np.uint16)
        images += [image]
        
        # Get boundaries of the image
        x_min = my_ds.x.values.min()
        x_max = my_ds.x.values.max()
        y_min = my_ds.y.values.min()
        y_max = my_ds.y.values.max()
        boundaries += [((x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min))]
        
    stacks = []
    for i in range(0, tiles):
        stack = np.hstack(images[i::tiles])
        stacks += [stack]

    composition = np.dstack(stacks)

    if to8bit:
        composition = np.array(composition >> 6, dtype=np.uint8)

    return composition, point,boundaries
