#### MAIN PIPELINE ####
from build_dataset import build_dataset
from run_model import run

### PARAMETERS ###
image_size = 200
sample_size = 1

variable = "pred_inc_mean"
kind = "reg"
model = "small_cnn"

# Step 1: Run Pansharpening in QGIS to get the images in high resolution
# Step 2: Run build_dataset to generate the dataset (npy files and metadata.csv)
build_dataset(image_size=image_size, sample_size=sample_size, variable=variable)

# Step 3: Train the Model
run(
    model_name=model,
    pred_variable=variable,
    kind=kind,
    small_sample=True,
    weights="imagenet",
    im_size=image_size
)
