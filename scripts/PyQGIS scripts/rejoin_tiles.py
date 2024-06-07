import os
import sys

sys.path.append("..")  # Add the parent directory to the sys.path
import build_dataset
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
import os

path_satelites = rf"\\wsl.localhost\UbuntuE\home\nico\data"

# # for capture_id, files in files_by_id.items():
# capture_id = "202211031357171"
# files = os.listdir(rf"\\wsl.localhost\UbuntuE\home\nico\data\2022")
# files = [f for f in files if f.endswith(".tif")]
# id_files = [
#     rf"\\wsl.localhost\UbuntuE\home\nico\data\2022\{f}"
#     for f in files
#     if capture_id in f
# ]
# id_files_matrix = build_dataset.generate_matrix_of_files(id_files)

# ds = xr.open_mfdataset(
#     id_files_matrix[0],
#     combine="nested",
#     concat_dim=["y"],
#     engine="rasterio",
#     parallel=True,
#     chunks={"x": 1000, "y": 1000, "band": 1},
# )
# ds

for year in [2022, 2018, 2013]:
    print(year)
    files = os.listdir(rf"{path_satelites}/{year}")
    files = [f for f in files if f.endswith(".tif")]
    capture_ids = set([f.split("_")[1] for f in files])

    assert all([os.path.isfile(rf"{path_satelites}/{year}/{f}") for f in files])

    files_by_id = {}
    datasets = {}
    for capture_id in capture_ids:
        print(f"Construyendo dataset: {capture_id}")
        # for capture_id, files in files_by_id.items():
        id_files = [rf"{path_satelites}/{year}/{f}" for f in files if capture_id in f]
        id_files_matrix = build_dataset.generate_matrix_of_files(id_files)
        ds = xr.open_mfdataset(
            id_files_matrix,
            combine="nested",
            concat_dim=["x", "y"],
            engine="rasterio",
            parallel=True,
        )

        with ProgressBar():
            ds.to_netcdf(
                rf"E:\{year}\pansharpened_{capture_id}.nc",
                encoding={
                    "band_data": {
                        "dtype": np.uint8,
                        "zlib": True,
                        "complevel": 7,
                        "chunksizes": (1, 1000, 1000),
                    }
                },
            )
