#### MAIN PIPELINE ####
from build_dataset import build_dataset
from run_model import run

# TODO:
## hay imagenes que se generan con un borde negro porque asi vienen en la data orignial. Hay que crpoear eso!
## Visualizar data augmentation

### PARAMETERS ###
image_size = 500
sample_size = 1
resizing_size = 200
tiles = 1

variable = "ln_pred_inc_mean"
kind = "reg"
model = "small_cnn"

# Step 1: Run Pansharpening in QGIS to get the images in high resolution
# Step 2: Run build_dataset to generate the dataset (npy files and metadata.csv)
# build_dataset(
#     image_size=image_size, sample_size=sample_size, variable=variable, tiles=tiles
# )

# Step 3: Train the Model
run(
    model_name=model,
    pred_variable=variable,
    kind=kind,
    small_sample=False,
    # weights="imagenet",
    image_size=image_size,
    sample_size=sample_size,
    resizing_size=resizing_size,
)
