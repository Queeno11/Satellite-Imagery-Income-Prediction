from run_model import run
import json
import os

# params_path = r"/mnt/d/Maestría/Tesis/Repo/data/data_in/Model parameters"
# params = os.listdir(params_path)

# for param_txt in params:
#     with open(os.path.join(params_path, param_txt), "r") as file:
#         content = file.read()

#     # Convert the content into a dictionary
#     params = json.loads(content)

params = {
    "model_name": "effnet_v2S",
    "kind": "reg",
    "weights": None,
    "image_size": 128,
    "resizing_size": 128,
    "tiles": 1,
    "nbands": 4,
    "stacked_images": [1, 4],
    "sample_size": 5,
    "small_sample": False,
    "n_epochs": 150,
    "learning_rate": 0.0001,
    "sat_data": "pleiades",
    "years": [2013, 2018, 2022],
    "extra": "",
}
run(params, train=False, compute_loss=True, generate_grid=False)
