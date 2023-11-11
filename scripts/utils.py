import shapely
import numpy as np
import geopandas as gpd
import cv2
import skimage


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


def point_column_to_x_y(df):
    df.point = df.point.str.replace("\(|\)", "", regex=True).str.split(",")
    df["x"] = df.point.str[0]
    df["y"] = df.point.str[1]
    return df[["x", "y"]]


def image_from_point(ds, point, max_bias=0, tile_size=100):
    x, y = point
    # Identifico el raster más cercano a este punto -- va a ser el centro de la imagen
    idx_x, idx_y = find_nearest_raster(ds.x, ds.y, x, y, max_bias=max_bias)

    # # Genero el cuadrado que quiero capturar en la imagen
    idx_x_min = round(idx_x - tile_size / 2)
    idx_x_max = round(idx_x + tile_size / 2)
    idx_y_min = round(idx_y - tile_size / 2)
    idx_y_max = round(idx_y + tile_size / 2)

    # If any of the indexes are negative, move to the next iteration
    if (
        (idx_x_min < 0)
        | (idx_x_max > ds.x.size)
        | (idx_y_min < 0)
        | (idx_y_max > ds.y.size)
    ):
        image = np.zeros((4, 0, 0))
        return image

    # Filtro el dataset para quedarme con esWa imagen
    image_dataset = ds.isel(
        x=slice(idx_x_min, idx_x_max), y=slice(idx_y_min, idx_y_max)
    )

    return image_dataset


def get_image_bounds(image_dataset, boundaries, previous_total_boundaries=None):
    if previous_total_boundaries is None:
        min_x_min, max_x_max, min_y_min, max_y_max = +999, -999, +999, -999
    else:
        min_x_min, max_x_max, min_y_min, max_y_max = previous_total_boundaries

    # Get boundaries of the image
    x_min = image_dataset.x.values.min()
    x_max = image_dataset.x.values.max()
    y_min = image_dataset.y.values.min()
    y_max = image_dataset.y.values.max()
    boundaries += [((x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min))]

    # Get minumum and maximum values for each axis of all the images
    if x_min < min_x_min:
        min_x_min = x_min
    if x_max > max_x_max:
        max_x_max = x_max
    if y_min < min_y_min:
        min_y_min = y_min
    if y_max > max_y_max:
        max_y_max = y_max
    total_boundaries = (min_x_min, max_x_max, min_y_min, max_y_max)

    return boundaries, total_boundaries

def assert_image_is_valid(image, n_bands, size):
    ''' Check whether image is valid. For using after image_from_point.
        If the image is valid, returns the image in numpy format. 
    '''
    try:
        image = image_dataset.band_data
        is_valid = True
    except:
        is_valid = False

    return image, is_valid

def random_image_from_census_tract(
    ds,
    icpag,
    link,
    start_point=None,
    size=100,
    n_bands=4,
    n_stacked_images=1,
    to8bit=True,
    image_return_only=False,
):
    """Genera una imagen aleatoria de tamaño size centrada en un punto aleatorio del radio censal {link}.

    Parameters:
    -----------
    ds: xarray.Dataset, dataset con las imágenes de satélite
    icpag: geopandas.GeoDataFrame, shapefile con los radios censales
    link: str, 9 dígitos que identifican el radio censal
    start_point: tuple, coordenadas del punto inicial, en formato (x, y). Si se establece,
        la imagen de la tile 0 se genera con este punto como centro. Si no,
        se genera un punto aleatorio.
    tiles: int, cantidad de imágenes a generar por lado
    size: int, tamaño de la imagen a generar, en píxeles
    bias: int, cantidad de píxeles que se mueve el punto aleatorio de las tiles
    to8bit: bool, si es True, convierte la imagen a 8 bits

    Returns:
    --------
    image: numpy.ndarray or None. numpy.ndarray, imagen de tamaño size x size. Si no se encuentra la imagen, devuelve None.
    point: tuple, coordenadas del punto seleccionado

    """
    total_bands = n_bands * n_stacked_images
    is_valid = False
    counter = 0

    while (is_valid==False) & (counter <= 2):
        # Get start point
        if start_point is None:
            x, y = random_point_from_geometry(
                icpag.loc[icpag["link"] == link], size
            )
            point = (x[0], y[0])
        else:
            point = start_point

        ## Generate images
        # big
        big_image_dataset = image_from_point(
            ds, point, tile_size=size*4
        )
        big_image, big_is_valid = assert_image_is_valid(big_image_dataset, n_bands, size)

        if big_is_valid is False:
            counter += 1
            continue

        # small
        small_image_dataset = image_from_point(
            ds, point, tile_size=size
        )
        small_image, small_is_valid = assert_image_is_valid(small_image_dataset, n_bands, size)
        if small_is_valid is False:
            counter += 1
            continue        

    if small_is_valid & big_is_valid:
        image = np.concatenate([small_image, big_image], axis=0) # Concat over bands
        assert image.shape == (bands, size, size)
        
        # Si la imagen tiene el tamaño correcto, la guardo
        image = image.to_numpy().astype(np.uint16)

        # Get image bounds and update total boundaries
        boundaries, total_boundaries = get_image_bounds(
            big_image_dataset, boundaries, total_boundaries
        )

        if to8bit:
            image = np.array(image >> 6, dtype=np.uint8)

    else:
        # print("Some tiles were not found. Image not generated...")
        image = None
        point = (None, None)
        boundaries = None
        total_boundaries = (None, None, None, None)

    return image, point, boundaries, total_boundaries


def random_tiled_image_from_census_tract(
    ds,
    icpag,
    link,
    start_point=None,
    tiles=1,
    size=100,
    bias=4,
    to8bit=True,
    image_return_only=False,
):
    """Genera una imagen aleatoria de tamaño size centrada en un punto aleatorio del radio censal {link}.

    Parameters:
    -----------
    ds: xarray.Dataset, dataset con las imágenes de satélite
    icpag: geopandas.GeoDataFrame, shapefile con los radios censales
    link: str, 9 dígitos que identifican el radio censal
    start_point: tuple, coordenadas del punto inicial, en formato (x, y). Si se establece,
        la imagen de la tile 0 se genera con este punto como centro. Si no,
        se genera un punto aleatorio.
    tiles: int, cantidad de imágenes a generar por lado
    size: int, tamaño de la imagen a generar, en píxeles
    bias: int, cantidad de píxeles que se mueve el punto aleatorio de las tiles
    to8bit: bool, si es True, convierte la imagen a 8 bits

    Returns:
    --------
    image: numpy.ndarray or None. numpy.ndarray, imagen de tamaño size x size. Si no se encuentra la imagen, devuelve None.
    point: tuple, coordenadas del punto seleccionado

    """
    images = []
    boundaries = []
    total_boundaries = None

    tile_size = size // tiles
    tiles_generated = 0
    for tile in range(0, tiles**2):
        image = np.zeros((4, 0, 0))
        counter = 0

        while (image.shape != (4, tile_size, tile_size)) & (counter <= 2):
            if tile == 0:
                max_bias = 0
                if start_point is None:
                    x, y = random_point_from_geometry(
                        icpag.loc[icpag["link"] == link], tile_size
                    )
                    point = (x[0], y[0])
                else:
                    point = start_point
            else:
                max_bias = bias * size
                # Obtengo un punto aleatorio del radio censal con un buffer de tamaño size
                x, y = random_point_from_geometry(
                    icpag.loc[icpag["link"] == link], tile_size
                )
                point = (x[0], y[0])

            image_dataset = image_from_point(
                ds, point, max_bias=max_bias, tile_size=tile_size
            )
            
            try:
                image = image_dataset.band_data
            except:
                image = np.zeros((4, 0, 0))
            counter += 1

            if image.shape == (4, tile_size, tile_size):
                # Si la imagen tiene el tamaño correcto, la guardo y salgo del loop
                images += [image.to_numpy().astype(np.uint16)]

                # Get image bounds and update total boundaries
                boundaries, total_boundaries = get_image_bounds(
                    image_dataset, boundaries, total_boundaries
                )

                tiles_generated += 1

    # Check if all the tiles were found
    if tiles_generated == tiles**2:
        # print("All tiles found")

        stacks = []
        for i in range(0, tiles):
            stack = np.hstack(images[i::tiles])
            stacks += [stack]

        composition = np.dstack(stacks)

        if to8bit:
            composition = np.array(composition >> 6, dtype=np.uint8)

    else:
        # print("Some tiles were not found. Image not generated...")
        composition = None
        point = (None, None)
        boundaries = None
        total_boundaries = (None, None, None, None)

    # if image_return_only:
    #     return composition

    return composition, point, boundaries, total_boundaries



def process_image(img, resizing_size):
    img = np.moveaxis(
        img, 0, 2
    )  # Move axis so the original [4, 512, 512] becames [512, 512, 4]
    # img = img[:, :, :3]  # FIXME: remove this line when using 4 channels
    image_size = img.shape[0]

    if image_size != resizing_size:
        img = cv2.resize(
            img, dsize=(resizing_size, resizing_size), interpolation=cv2.INTER_CUBIC
        )
    img = (
        skimage.exposure.equalize_hist(img) * 255
    )  # FIXME: ¿equalizar imagen por imagen o el tileset entero?
    img = img.astype(np.uint8)

    return img


def augment_image(img):
    rand_2 = np.random.randint(-10, 10) / 100  # Random number between -0.1 and 0.1

    # Random flip
    if np.random.rand() > 0.5:
        img = np.fliplr(img)
    if np.random.rand() > 0.5:
        img = np.flipud(img)

    # Adjust contrast (power law transformation)
    #   see: https://web.ece.ucsb.edu/Faculty/Manjunath/courses/ece178W03/EnhancePart1.pdf
    rand_gamma = (
        np.random.randint(40, 250) / 100
    )  # Random number between 0.4 and 2.5 (same level of contrast)
    img = skimage.exposure.adjust_gamma(img, gamma=rand_gamma)

    # Rescale intensiry
    rand_min = np.random.randint(0, 5) / 10  # Random number between 0 and 0.5
    rand_max = np.random.randint(0, 5) / 10  # Random number between 0 and 0.5
    v_min, v_max = np.percentile(img, (rand_min, 100 - rand_max))
    img = skimage.exposure.rescale_intensity(img, in_range=(v_min, v_max))

    return img
