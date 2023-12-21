import processing

import os
from os.path import dirname

# Parameters for the algorithm
images = [
    "IMG_PHR1A_type_201302051412270_ORT_6741387101-tnumber_region.TIF",
    "IMG_PHR1A_type_201302071358055_ORT_6741390101-tnumber_region.TIF",
    "IMG_PHR1A_type_201302071358259_ORT_6741392101-tnumber_region.TIF",
]
r_values = [1, 2, 3]
c_values = [1, 2]
types = [("P", "1"), ("MS", "2")]
path = r"D:\Imagenes Satelitales\2013"

# Get the full list of files
panchromatic = []
multispectral = []
outputs = []

for img in images:
    for r in r_values:
        for c in c_values:
            for type, typenumber in types:
                filename = (
                    img.replace("type", type)
                    .replace("tnumber", typenumber)
                    .replace("region", f"R{r}C{c}")
                )
                type_folder = f"IMG_PHR1A_{type}_00{typenumber}"
                satellite_pic_number = img.split("_")[5].split("-")[0]

                full_path = os.path.join(
                    path, satellite_pic_number, type_folder, filename
                )
                assert os.path.exists(full_path)

                if type == "P":
                    panchromatic += [full_path]
                elif type == "MS":
                    multispectral += [full_path]

            output_name = f"pansharpened_{satellite_pic_number}_R{r}C{c}.tif"
            output_folder = dirname(dirname(full_path))
            output_path = os.path.join(output_folder, output_name)
            outputs += [output_path]

assert len(panchromatic) == len(multispectral) == len(outputs)

i=0
n_imgs = len(panchromatic)
for p, ms, out in zip(panchromatic, multispectral, outputs):
    processing.run(
        "gdal:pansharp",
        {
            "SPECTRAL": ms,
            "PANCHROMATIC": p,
            "OUTPUT": out,
            "RESAMPLING": 2,
            "OPTIONS": "COMPRESS=NONE|BIGTIFF=IF_NEEDED",
            "EXTRA": "",
        },
    )
    print(f"{out} created. {n_imgs-i} images remains.")
    i += 1
