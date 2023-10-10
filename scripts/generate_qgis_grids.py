import processing
from qgis.core import QgsCoordinateReferenceSystem

import os
from os.path import dirname

# Parameters for the algorithm
spacing128px = 0.00059
params = {
    "size128_tiles1": spacing128px,
    "size128_tiles2": spacing128px / 2,
    "size256_tiles1": spacing128px * 2,
    "size512_tiles1": spacing128px * 4,
}

for name, spacing in params.items():
    processing.run(
        "native:creategrid",
        {
            "TYPE": 2,
            "EXTENT": "-58.903973939,-58.127860744,-34.971389075,-34.204721477 [EPSG:4326]",
            "HSPACING": spacing,
            "VSPACING": spacing,
            "HOVERLAY": 0,
            "VOVERLAY": 0,
            "CRS": QgsCoordinateReferenceSystem("EPSG:4326"),
            "OUTPUT": f"D:/Maestría/Tesis/Repo/data/data_in/Grillas/grid_{name}.parquet",
        },
    )
