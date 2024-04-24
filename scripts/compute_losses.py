from run_model import run
import json
import os

params_path = r"/mnt/d/Maestría/Tesis/Repo/data/data_in/Model parameters"
params = os.listdir(params_path)

for param_txt in params:
    with open(os.path.join(params_path, param_txt), "r") as file:
        content = file.read()

    # Convert the content into a dictionary
    params = json.loads(content)

    run(params, train=False, compute_loss=True, generate_grid=False)
