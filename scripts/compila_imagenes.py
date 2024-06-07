import os
import xarray as xr
from dask.diagnostics import ProgressBar


for year in [2013, 2018, 2022]:

    path = rf"/home/nico/data/{year}"
    files = os.listdir(path)
    capture_ids = set([f.split("_")[1] for f in files])

    for capture_id in capture_ids:
        print(f"Running capture {capture_id} - {year}")
        dss = []
        for f in files:
            if (capture_id in f) and (f.endswith(".tif")):
                dss += [
                    xr.open_dataset(os.path.join(path, f), chunks={"x": 500, "y": 500})
                ]

        merged = xr.combine_by_coords(dss)

        with ProgressBar():
            merged_computed = merged.compute()

        merged_computed.to_netcdf(
            os.path.join(path, f"full_{capture_id}.nc"), compute=False
        )
