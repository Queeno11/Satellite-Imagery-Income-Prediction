import processing

import os
from os.path import dirname

# Parameters for the algorithm
# Example name: IMG_PNEO3_202211011409151_MS-FS_ORT_PWOI_000047717_1_22_F_5_R5C2.tif
path = r"E:\2022 imagenes"
output_folder = r"D:\Maestría\Tesis\Repo\data\data_in\Pansharpened"
files = os.listdir(path)
ms_files = [f for f in files if f.endswith(".TIF") and "_MS-FS_" in f]

multispectral = [os.path.join(path, f) for f in ms_files]
panchromatic = [f.replace("_MS-FS_", "_PAN_").replace("_P_", "") for f in multispectral]

assert len(panchromatic) == len(multispectral)

i = 0
n_imgs = len(panchromatic)
for p, ms in zip(panchromatic, multispectral):
    satellite_pic_number = ms.split("_")[2]
    ms_filename = os.path.split(ms)[-1]
    region = ms_filename.split("_")[-1]  # example: R5C2.tif
    r = region[1]
    c = region[3]
    output_name = f"pansharpened_{satellite_pic_number}_R{r}C{c}.tif"
    out = os.path.join(output_folder, output_name)

    processing.run(
        "gdal:pansharp",
        {
            "SPECTRAL": ms,
            "PANCHROMATIC": p,
            "OUTPUT": out,
            "RESAMPLING": 2,
            "OPTIONS": "COMPRESS=DEFLATE|PREDICTOR=2",
            "EXTRA": "",
        },
    )
    print(f"{out} created. {n_imgs-i} images remains.")
    i += 1
