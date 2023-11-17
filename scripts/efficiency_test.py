import xarray as xr
import geopandas as gpd
import numpy as np
from shapely import Point
import build_dataset
import utils 

img_size = 128
print("Reading dataset...")
sat_imgs_datasets, extents = build_dataset.load_satellite_datasets()
df = build_dataset.load_icpag_dataset()
df = build_dataset.assign_links_to_datasets(df, extents, verbose=True)
print("Dataset loaded!")

def random_point_from_geometry(polygon, size=100):
    '''Generates a random point within the bounds of a Polygon.'''

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

def get_image_bounds(image_dataset, boundaries=[], previous_total_boundaries=None):
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

def find_nearest_idx(array, value):
    ''' Returns the index of the nearest value of {value} from the array'''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def image_from_point(dataset, point, img_size=128):
    
    # Find the rearest raster of this random point
    x, y = point
    idx_x = np.searchsorted(dataset.x, x, side='left', sorter=None)
    idx_y = np.searchsorted(-dataset.y, -y, side='left', sorter=None) # The y array is inverted! https://stackoverflow.com/questions/43095739/numpy-searchsorted-descending-order
    
    # Create the indexes of the box of the image
    idx_x_min = round(idx_x - img_size / 2)
    idx_x_max = round(idx_x + img_size / 2)
    idx_y_min = round(idx_y - img_size / 2)
    idx_y_max = round(idx_y + img_size / 2)

    # If any of the indexes are negative, move to the next iteration
    if (
        (idx_x_min < 0)
        | (idx_x_max > dataset.x.size)
        | (idx_y_min < 0)
        | (idx_y_max > dataset.y.size)
    ):
        image = np.zeros(shape=(1,1,1))
        return image

    
    image = dataset.isel(
        x=slice(idx_x_min, idx_x_max),
        y=slice(idx_y_min, idx_y_max)
    )
    
    image = image.band_data
    return image

def random_image_from_census_tract(dataset, polygon):
    # Sample point from the polygon's box
    point = random_point_from_geometry(polygon)
    image_da = image_from_point(dataset, point)    
    return image_da

def stacked_image_from_census_tract(dataset, polygon, img_size=100, n_bands=4, stacked_images=[1,3]):
    images_to_stack = []
    boundaries = []
    total_bands = n_bands*len(stacked_images)

    # Sample point from the polygon's box
    point = random_point_from_geometry(polygon)
    
    for size_multiplier in stacked_images:
        image_size = img_size*size_multiplier
        image_da = image_from_point(dataset, point, image_size)
        try:   
            image = image_da.to_numpy()[:,::size_multiplier,::size_multiplier]
            image = image.astype(np.uint16)
            images_to_stack += [image]
        except:
            image = np.zeros(shape=(1,1,1))
            bounds = None
            return image, bounds

    # Get total bounds
    bounds = get_image_bounds(image_da) # The last image has to be the bigger
    image = np.concatenate(images_to_stack, axis=0) # Concat over bands
    assert image.shape == (total_bands, img_size, img_size)
    
    return image, bounds


def get_image_bounds(image_ds):
    # The array y is inverted!
    min_x = image_ds.x[0].item()
    min_y = image_ds.y[-1].item()
    max_x = image_ds.x[-1].item()
    max_y = image_ds.y[0].item()
    
    return min_x, min_y, max_x, max_y
    
# def stacked_images_from_point(ds, point, size_small=100, n_bands=4, stacked_images=[1,2]):
#     ''' Generate image composition of many stacked images along the band axis (first axis). 
#         The resulting image is, for example, an array of size (8, 128, 128), where the first 4
#         bands represent the first image.       
#     '''
#     images_to_stack = []
#     boundaries = []
#     total_bands = n_bands*len(stacked_images)

#     for size_multiplier in stacked_images:

#         image_size = size_small*size_multiplier
#         image_dataset = image_from_point(
#             ds, point, tile_size=image_size
#         )
#         image, image_is_valid = assert_image_is_valid(image_dataset, n_bands, image_size)

#         if image_is_valid is False:
#             image = None
#             is_valid = False
#             boundaries = None
#             total_boundaries = None
#             return image, is_valid, boundaries, total_boundaries # Return as soon as some error apears to save time

#         else:
#             is_valid = True
#             boundaries, total_boundaries = get_image_bounds(
#                 image_dataset, boundaries
#             )
#             # Resize image to match size_small
#             image = image.to_numpy().astype(np.uint16)[:,::size_multiplier,::size_multiplier]
#             images_to_stack += [image]
            
#     # return images_to_stack, True
#     image = np.concatenate(images_to_stack, axis=0) # Concat over bands
#     assert image.shape == (total_bands, size_small, size_small)
#     is_valid = True
    
#     return image, is_valid, boundaries, total_boundaries


def get_data(i):
    
    # Get the polygon and the label of that index (neighbour)
    polygon = df.iloc[i]["geometry"]
    dataset = sat_imgs_datasets[df.iloc[i]["dataset"]]

    image, boundaries = stacked_image_from_census_tract(dataset, polygon, 128)
    label = df.iloc[i]["var"]
    
    if image.shape == (8, img_size, img_size):

        # Assert that data corresponds to train or test
        is_correct_type = build_dataset.assert_train_test_datapoint(
            boundaries, wanted_type="train"
        )

        if is_correct_type == False:  # If the point is not train/test, discard it
            image = np.zeros(shape=(4,0,0))
            value = np.nan
            return image, label
    
        # Reduce quality and process image 
        image = (image >> 6)
        image = utils.process_image(image, resizing_size=128)

        # Augment dataset
        # if type == "train":
        image = utils.augment_image(image)

    else:
        image = np.zeros(shape=(1,1,1))
        value = np.nan
    
    return image, label

if __name__ == "__main__":

    import run_model
    import tensorflow as tf
    tast = run_model.get_data(tf.constant(1500), 1, "train", load=True)
    # from tqdm import tqdm
    # andan = 0
    # no_andan = 0
    # for i in tqdm(range(0, 3000)):
    #     image, val = get_data(i)
    #     if image.shape == (img_size, img_size, 8):
    #         andan += 1
    #     else:
    #         no_andan += 1

    # print(f"Train: {andan}\n No andan: {no_andan}")

