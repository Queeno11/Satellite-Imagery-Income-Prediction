import processing
from qgis.core import QgsCoordinateReferenceSystem

# Parameters for the algorithm
images = [
    r"pansharpened_6741392101_R3C2.tif",
    r"pansharpened_6741387101_R1C1.tif",
    r"pansharpened_6741387101_R1C2.tif",
    r"pansharpened_6741387101_R2C1.tif",
    r"pansharpened_6741387101_R2C2.tif",
    r"pansharpened_6741387101_R3C1.tif",
    r"pansharpened_6741387101_R3C2.tif",
    r"pansharpened_6741390101_R1C1.tif",
    r"pansharpened_6741390101_R1C2.tif",
    r"pansharpened_6741390101_R2C1.tif",
    r"pansharpened_6741390101_R2C2.tif",
    r"pansharpened_6741390101_R3C1.tif",
    r"pansharpened_6741390101_R3C2.tif",
    r"pansharpened_6741392101_R1C1.tif",
    r"pansharpened_6741392101_R1C2.tif",
    r"pansharpened_6741392101_R2C1.tif",
    r"pansharpened_6741392101_R2C2.tif",
    r"pansharpened_6741392101_R3C1.tif"
]
path_in = r"d:\Maestría\Tesis\Repo\data\data_in\Pansharpened\2013"
path_out = r"d:\Maestría\Tesis\Repo\data\data_in\Compressed\2013"

n_imgs = len(images)
i=0
for image in images:
    processing.run(
        "gdal:translate", {
            'INPUT': f'{path_in}/{image}',
            'TARGET_CRS':QgsCoordinateReferenceSystem('EPSG:4326'),
            'NODATA':None,
            'COPY_SUBDATASETS':False,
            'OPTIONS':'COMPRESS=JPEG|JPEG_QUALITY=75|TILED=Yes',
            'EXTRA':'-scale_1 0 2577 -scale_2 0 2419 -scale_3 0 2036 -scale_4 0 3983',
            'DATA_TYPE':1,
            'OUTPUT':f'{path_out}/{image}'
         })    
    print(f"{path_out}/{image} created. {n_imgs-i} images remains.")
    i += 1
