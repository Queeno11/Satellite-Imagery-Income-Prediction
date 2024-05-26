import processing
from qgis.core import QgsRasterLayer

import os
from os.path import dirname


## NOTA: QGIS no tiene forma (que conozca) de limpiar los TEMPORARY_OUTPUTS (no funciona garbace collect ni nada de eso), así que
#   se rompe cuando el disco C está completo... Ir corriendo de a pedazos. AL finalizar limpiar tempfiles

# Parameters for the algorithm
sat_captures = [
    "000058605_1_4_STD_A",
]

path = r"F:\Imagenes Satelitales\2022"

# Get the full list of files

import os

for sat_capture in sat_captures:
    try:
        folder = rf"F:\Imagenes Satelitales\2022\{sat_capture}\IMG_01_PNEO4_MS-FS"
        files = os.listdir(folder)
        rgb_files = [f for f in files if f.endswith(".TIF") and "_RGB_" in f]
    except:
        folder = rf"F:\Imagenes Satelitales\2022\{sat_capture}\IMG_01_PNEO3_MS-FS"
        files = os.listdir(folder)
        rgb_files = [f for f in files if f.endswith(".TIF") and "_RGB_" in f]

    print(f"Transformando {sat_capture}")
    rgb_files = [os.path.join(folder, f) for f in rgb_files]
    ned_files = [f.replace("_RGB_", "_NED_") for f in rgb_files]
    out_files = [f.replace("_RGB_", "_") for f in rgb_files]
    
    assert len(rgb_files) == len(ned_files) == len(out_files)

    i = 0
    n_imgs = len(rgb_files)
    for rgb, ned, out in zip(rgb_files, ned_files, out_files):
        out = os.path.split(out)[-1]
        out = rf"E:\2022 imagenes\{out}"
        if os.path.isfile(out):
            continue
            
        # Step 1: Extract NIR band using gdal:translate
        result_extract_nir = processing.run(
            "gdal:translate",
            {
                "INPUT": ned,
                "EXTRA": "-b 1",
                "DATA_TYPE": 0,  # Same data type as source
                "OUTPUT": "TEMPORARY_OUTPUT",
            },
        )
        nir_only = QgsRasterLayer(result_extract_nir["OUTPUT"], "Extracted NIR")

        result_merge_rgb_nir = processing.run(
            "gdal:merge",
            {
                "INPUT": [
                    rgb,
                    nir_only,
                ],
                "PCT": False,
                "SEPARATE": True,
                "NODATA_INPUT": None,
                "NODATA_OUTPUT": None,
                "OPTIONS": "COMPRESS=NONE|BIGTIFF=IF_NEEDED",
                "EXTRA": "",
                "DATA_TYPE": 2,
                "OUTPUT": out,
            },
        )

        print(f"{out} created. {n_imgs-i} images remains.")
        i += 1
