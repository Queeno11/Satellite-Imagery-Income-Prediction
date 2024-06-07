import cv2
import shapely
import numpy as np
import geopandas as gpd
import xarray as xr
import skimage
from skimage import color
from shapely import Point
import build_dataset


def get_dataset_extent(ds, image_size=None):
    """Return a polygon with the extent of the dataset

    Params:
    ds: xarray dataset

    Returns:
    polygon: shapely polygon with the extent of the dataset
    """

    if image_size is not None:
        limit = int(image_size)
        ds = ds.isel(x=slice(limit, -limit), y=slice(limit, -limit))

    x_min = ds.x.min()
    x_max = ds.x.max()
    y_min = ds.y.min()
    y_max = ds.y.max()

    bbox = (x_min, y_min, x_max, y_max)

    # Turn bbox into a shapely polygon
    polygon = shapely.geometry.box(*bbox)

    return polygon


def get_datasets_for_polygon(poly, extents):
    """Devuelve el nombre del dataset que contiene el polígono seleccionado."""

    correct_datasets = []
    for name, extent in extents.items():
        if extent.intersects(poly):
            correct_datasets += [name]

    if len(correct_datasets) == 0:
        print("Ningun dataset contiene al polígono seleccionado.")

    return correct_datasets


def random_point_from_geometry(polygon, size=100):
    """Generates a random point within the bounds of a Polygon."""

    # Get bounds of the shapefile's polygon
    (minx, miny, maxx, maxy) = polygon.bounds

    # Loop until finding a random point inside the polygon
    while 0 == 0:
        # generate random data within the bounds
        x = np.random.uniform(minx, maxx, 1)
        y = np.random.uniform(miny, maxy, 1)
        point = Point(x, y)
        if polygon.contains(point):
            return x[0], y[0]


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
    x_idx = np.searchsorted(x_array, x_value, side="left", sorter=None) + x_bias
    y_idx = np.searchsorted(y_array, y_value, side="left", sorter=None) + y_bias
    return x_idx, y_idx


def point_column_to_x_y(df):
    df.point = df.point.str.replace(r"\(|\)", "", regex=True).str.split(",")
    df["x"] = df.point.str[0]
    df["y"] = df.point.str[1]
    return df[["x", "y"]]


def find_index_of_point_in_dataset(dataset, point):
    # Find the rearest raster of this random point
    x, y = point
    idx_x = np.searchsorted(dataset.x, x, side="left", sorter=None)
    idx_y = np.searchsorted(
        -dataset.y, -y, side="left", sorter=None
    )  # The y array is inverted! https://stackoverflow.com/questions/43095739/numpy-searchsorted-descending-order
    return idx_x, idx_y


def filter_datasets_by_idx_and_buffer(
    dataset, idx_x, idx_y, img_size, validate_size=True
):
    # Create the indexes of the box of the image
    idx_x_min = round(idx_x - img_size / 2)
    idx_x_max = round(idx_x + img_size / 2)
    idx_y_min = round(idx_y - img_size / 2)
    idx_y_max = round(idx_y + img_size / 2)

    # If any of the indexes are negative, move to the next iteration
    if validate_size:
        if (
            (idx_x_min < 0)
            | (idx_x_max > dataset.x.size)
            | (idx_y_min < 0)
            | (idx_y_max > dataset.y.size)
        ):
            return None
    else:
        if idx_x_min < 0:
            idx_x_min = 0
        if idx_y_min < 0:
            idx_y_min = 0

    filtered = dataset.isel(
        x=slice(idx_x_min, idx_x_max), y=slice(idx_y_min, idx_y_max)
    )

    return filtered


def image_from_point(dataset, point, img_size=128):

    # Find the rearest raster of this random point
    idx_x, idx_y = find_index_of_point_in_dataset(dataset, point)

    image = filter_datasets_by_idx_and_buffer(dataset, idx_x, idx_y, img_size)

    if image is None:
        return np.zeros(shape=(1, 1, 1))

    image = image.band_data

    if image.chunks is not None:
        # If the image is not computed, compute it
        image = image.compute()

    return image


def dataset_for_image_at_bound(datasets, point, img_size=512):

    minumum_valid_shape = (1, img_size, img_size)

    if img_size > 512:
        print(
            "Warning. The image size is bigger than 512. The buffer of dataset_for_image_at_bound() should be changed..."
        )
    # Lista de los capture_ids
    ids = [f.encoding["source"].split("\\")[-1].split("_")[1] for f in datasets]
    ids = list(set(ids))

    # Itero por lista de capture_ids
    for capture_id in ids:
        # print(capture_id)
        composition = []
        datasets_id = [d for d in datasets if capture_id in d.encoding["source"]]
        # Selecciona y filtra datasets que tienen el punto
        for dataset in datasets_id:

            # Find the rearest raster of this random point
            idx_x, idx_y = find_index_of_point_in_dataset(dataset, point)
            # print(idx_x, idx_y)
            filtered_ds = filter_datasets_by_idx_and_buffer(
                dataset, idx_x, idx_y, img_size, validate_size=False
            )
            # print(filtered_ds.band_data.shape)
            if (filtered_ds is None) or (0 in filtered_ds.band_data.shape):
                continue

            composition += [filtered_ds]

        # Build composed_dataset
        if len(composition) == 1:
            # Select dataset
            current_shape = composition[0].band_data.shape
            # Validate size
            minumum_valid_shape = (1, img_size, img_size)
            has_valid_shape = all(
                [(a >= b) for a, b in zip(current_shape, minumum_valid_shape)]
            )
            if has_valid_shape:
                composed_dataset = composition[0]
                return composed_dataset
            else:
                continue

        # Return an image if possible
        elif len(composition) > 1:
            # Select dataset
            try:
                composed_dataset = xr.combine_by_coords(
                    composition, combine_attrs="override"
                )

            except Exception as e:
                # print(e)
                composition = build_dataset.generate_matrix_of_datasets(composition)
                composed_dataset = xr.combine_nested(
                    composition, combine_attrs="override", concat_dim=["y", "x"]
                )
            # Validate size
            current_shape = composed_dataset.band_data.shape
            has_valid_shape = all(
                [(a >= b) for a, b in zip(current_shape, minumum_valid_shape)]
            )
            if has_valid_shape:
                # Sort so it matches the original format
                composed_dataset = composed_dataset.sortby("y", ascending=False)
                composed_dataset = composed_dataset.sortby("x", ascending=True)
                return composed_dataset
            else:
                continue
        else:
            continue


def get_image_bounds(image_ds):
    # The array y is inverted!
    min_x = image_ds.x[0].item()
    min_y = image_ds.y[-1].item()
    max_x = image_ds.x[-1].item()
    max_y = image_ds.y[0].item()

    return min_x, min_y, max_x, max_y


def assert_image_is_valid(image_dataset, n_bands, size):
    """Check whether image is valid. For using after image_from_point.
    If the image is valid, returns the image in numpy format.
    """
    try:
        image = image_dataset.band_data
        is_valid = True
    except:
        image = np.zeros(shape=[1, 100, 100])
        is_valid = False

    return image, is_valid


def stacked_image_from_census_tract(
    dataset,
    polygon,
    point=None,
    img_size=100,
    n_bands=4,
    stacked_images=[1, 3],
    bounds=True,
):

    images_to_stack = []
    total_bands = n_bands * len(stacked_images)

    if point is None:
        # Sample point from the polygon's box
        point = random_point_from_geometry(polygon)
        # point = polygon.centroid.x, polygon.centroid.y

    for size_multiplier in stacked_images:
        try:
            image_size = img_size * size_multiplier
            if type(dataset) == list:
                if len(dataset) > 1:
                    image_ds = dataset_for_image_at_bound(dataset, point, image_size)
                else:
                    image_ds = dataset[0]
            else:
                image_ds = dataset

            image_da = image_from_point(image_ds, point, image_size)
            image = image_da.to_numpy()[:n_bands, ::size_multiplier, ::size_multiplier]
            assert image.shape == (n_bands, img_size, img_size)

            image = np.nan_to_num(image)
            image = image.astype(np.uint8)
            images_to_stack += [image]

        except Exception as e:
            # print(e)
            image = np.zeros(shape=(n_bands, 1, 1))
            if bounds:
                return image, bounds
            return image

    # Get total bounds
    image = np.concatenate(images_to_stack, axis=0)  # Concat over bands
    assert image.shape == (total_bands, img_size, img_size)
    if bounds:
        bounds = get_image_bounds(image_da)  # The last image has to be the bigger
        return image, bounds

    return image


def random_image_from_census_tract(dataset, polygon, image_size):
    """Genera una imagen aleatoria de tamaño size centrada en un punto aleatorio del radio censal {link}.

    Parameters:
    -----------
    dataset: xarray.Dataset, dataset con las imágenes de satélite del link correspondiente
    polygon: shapely.Polygon, shape del radio censal
    image_size: int, tamaño del lado de la imagen

    Returns:
    --------
    image_da: xarray.DataArray, imagen de tamaño size x size. Si no se encuentra la imagen, devuelve una con size (8, size, size).
    point: tuple, coordenadas del punto seleccionado

    """

    # Sample point from the polygon's box
    point = random_point_from_geometry(polygon)
    image_da = image_from_point(dataset, point, img_size=image_size)
    return image_da


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
                images += [image.to_numpy().astype(np.uint8)]

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

        # if to8bit:
        #     composition = np.array(composition >> 6, dtype=np.uint8)

    else:
        # print("Some tiles were not found. Image not generated...")
        composition = None
        point = (None, None)
        boundaries = None
        total_boundaries = (None, None, None, None)

    # if image_return_only:
    #     return composition

    return composition, point, boundaries, total_boundaries


def process_image(img, resizing_size, moveaxis=True):
    if moveaxis:
        img = np.moveaxis(
            img, 0, 2
        )  # Move axis so the original [4, 512, 512] becames [512, 512, 4]
        image_size = img.shape[0]
    else:
        image_size = img.shape[2]

    if image_size != resizing_size:  # FIXME: Me las pasa a blanco y negro. Por qué?
        img = cv2.resize(
            img, dsize=(resizing_size, resizing_size), interpolation=cv2.INTER_CUBIC
        )

    return img


def augment_image(img):
    """
    Apply random augmentation to the input image. Image should have values in the 0-255 range.

    Randomly applies the following transformations to the input image:
    - Random flip horizontally or vertically with a 50% probability.
    - Adjusts contrast using power law transformation.
    - Rescales intensity by adjusting the pixel values within a random range.
    - Rescales saturation by adjusting the saturation values within a random range.

    Parameters:
    - img (numpy.ndarray): Input image as a NumPy array.

    Returns:
    - numpy.ndarray: Augmented image as a NumPy array.
    """

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

    # Rescale intensity
    rand_min = np.random.randint(0, 5) / 10  # Random number between 0 and 0.5
    rand_max = np.random.randint(0, 5) / 10  # Random number between 0 and 0.5
    v_min, v_max = np.percentile(img, (rand_min, 100 - rand_max))
    img = skimage.exposure.rescale_intensity(img, in_range=(v_min, v_max))

    # Rescale saturation
    rand_sat = 0.5 + np.random.rand()  # Random number between 0.5 and 1.5
    img_hsv = color.rgb2hsv(img)  # hue-saturation-value
    img_hsv[:, :, 1] *= rand_sat
    img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1], 0, 255)
    img = color.hsv2rgb(img_hsv)

    return img
