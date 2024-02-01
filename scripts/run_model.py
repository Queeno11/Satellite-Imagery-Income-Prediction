##############      Configuración      ##############
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from dotenv import dotenv_values

pd.set_option("display.max_columns", None)
envpath = "/mnt/d/Maestría/Tesis/Repo/scripts/globals.env"
if os.path.isfile(envpath):
    env = dotenv_values(envpath)
else:
    env = dotenv_values(r"D:/Maestría/Tesis/Repo/scripts/globals_win.env")

path_proyecto = env["PATH_PROYECTO"]
path_datain = env["PATH_DATAIN"]
path_dataout = env["PATH_DATAOUT"]
path_scripts = env["PATH_SCRIPTS"]
path_satelites = env["PATH_SATELITES"]
path_logs = env["PATH_LOGS"]
path_outputs = env["PATH_OUTPUTS"]
path_imgs = env["PATH_IMGS"]
# path_programas  = globales[7]
###############################################

import true_metrics
import custom_models
import build_dataset
import grid_predictions
import utils

import os
import sys
import scipy
import random
import pandas as pd
import xarray as xr
from typing import Iterator, List, Union, Tuple, Any
from datetime import datetime

# Mute TF low_level warnings: https://stackoverflow.com/questions/76912213/tf2-13-local-rendezvous-recv-item-cancelled
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import tensorflow_addons as tfa

from tensorflow.keras import layers, models, Model
from tensorflow.keras.callbacks import (
    TensorBoard,
    EarlyStopping,
    ModelCheckpoint,
    CSVLogger,
)
from tensorflow.keras.models import Sequential
import cv2
import skimage

# the next 3 lines of code are for my machine and setup due to https://github.com/tensorflow/tensorflow/issues/43174
try:
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print("No GPU set. Is the GPU already initialized?")


# Disable
def blockPrint():
    sys.__stdout__ = sys.stdout
    sys.stdout = open(os.devnull, "w")


# Restore
def enablePrint():
    sys.stdout.close()
    sys.stdout = sys.__stdout__


def create_train_test_dataframes(savename, sat_data="pleiades", small_sample=False):
    """Create train and test dataframes with the links and xr.datasets to use for training and testing

    Load the ICPAG dataset and assign the links to the corresponding xr.dataset, then split the census tracts
    into train and test. The train and test dataframes contain the links and xr.datasets to use for training and
    testing.
    """
    ### Open dataframe with files and labels
    print("Reading dataset...")
    df = build_dataset.load_icpag_dataset()
    all_years_datasets = {}

    if sat_data == "pleiades":
        years = [2013, 2018, 2022]
        for year in years:
            sat_imgs_datasets, extents = build_dataset.load_satellite_datasets(
                year=year
            )
            df = build_dataset.assign_datasets_to_gdf(
                df, extents, year=year, verbose=True
            )
            all_years_datasets[year] = sat_imgs_datasets
    elif sat_data == "landsat":
        sat_imgs_datasets, extents = build_dataset.load_landsat_datasets()
        df = build_dataset.assign_datasets_to_gdf(df, extents, year=2013, verbose=True)
        all_years_datasets[2013] = sat_imgs_datasets
    print("Datasets loaded!")

    print("Cleaning dataset...")
    # Clean dataframe and create datasets
    # df = df[df["sample"] <= 2]
    if small_sample:
        df = df.sample(2000, random_state=825).reset_index(drop=True)

    ### Split census tracts based on train/test
    #       (the hole census tract must be in the corresponding region)
    df["min_x"] = df.bounds["minx"]
    df["max_x"] = df.bounds["maxx"]
    df = build_dataset.split_train_test(df)
    df = df[
        ["link", "AREA", "var", "type", "geometry"]
        + [col for col in df.columns if "dataset" in col]
    ]

    ### Train/Test
    list_of_datasets = []

    df_test = df[df["type"] == "test"].copy().reset_index(drop=True)
    assert df_test.shape[0] > 0, f"Empty test dataset!"
    df_train = df[df["type"] == "train"].copy().reset_index(drop=True)
    assert df_train.shape[0] > 0, f"Empty train dataset!"

    df_test.to_feather(
        rf"{path_dataout}/test_datasets/{savename}_test_dataframe.feather"
    )
    print(
        "Se creó el archivo:",
        rf"{path_dataout}/test_datasets/{savename}_test_dataframe.feather",
    )
    df_train.to_feather(
        rf"{path_dataout}/train_datasets/{savename}_train_dataframe.feather"
    )
    print(
        "Se creó el archivo:",
        rf"{path_dataout}/train_datasets/{savename}_train_dataframe.feather",
    )

    return df_train, df_test, all_years_datasets


def create_datasets(
    df_train,
    df_test,
    all_years_datasets,
    image_size,
    resizing_size,
    sample=1,
    nbands=4,
    tiles=1,
    stacked_images=[1],
    savename="",
    save_examples=True,
):
    # Trying to replicate these good practices from here: https://cs230.stanford.edu/blog/datapipeline/#best-practices

    # Based on: https://medium.com/@acordier/tf-data-dataset-generators-with-parallelization-the-easy-way-b5c5f7d2a18
    def get_data(i, df_subset, type="train", load=False):
        # Decoding from the EagerTensor object. Extracts the number/value from the tensor
        #   example: <tf.Tensor: shape=(), dtype=uint8, numpy=20> -> 20
        i = i.numpy()

        if load:
            df_train, df_test, sat_imgs_datasets = create_train_test_dataframes(
                savename, small_sample=False
            )
        year = random.choices(
            population=list(all_years_datasets.keys()), weights=[1], k=1
        )[0]
        sat_img_dataset = all_years_datasets[year]
        # initialize iterators & params
        iteration = 0
        is_correct_type = False
        image = np.zeros(shape=(nbands, 0, 0))
        total_bands = nbands * len(stacked_images)
        img_correct_shape = (total_bands, image_size, image_size)

        # Get link, dataset and indicator value of the corresponding index
        polygon = df_subset.iloc[i]["geometry"]
        value = df_subset.iloc[i]["var"]
        link_dataset = sat_img_dataset[df_subset.iloc[i][f"dataset_{year}"]]

        while (is_correct_type == False) & (iteration <= 5):
            # Generate the image
            image, boundaries = utils.stacked_image_from_census_tract(
                dataset=link_dataset,
                polygon=polygon,
                img_size=image_size,
                n_bands=nbands,
                stacked_images=stacked_images,
            )

            # (1) Image has to have the correct shape
            if image.shape == img_correct_shape:
                # (2) Image has to fall in train or test side
                is_correct_type = build_dataset.assert_train_test_datapoint(
                    boundaries, wanted_type=type
                )

            iteration += 1

        if iteration >= 5:
            print(
                f"More than 5 interations for link {df_subset.iloc[i]['link']}, moving to next link..."
            )
            image = np.zeros(shape=(resizing_size, resizing_size, total_bands))
            value = 0
            return image, value

        # Reduce quality and process image
        image = utils.process_image(image, resizing_size=resizing_size)

        # Augment dataset
        if type == "train":
            image = utils.augment_image(image)
            # image = image

        # np.save(fr"/mnt/d/Maestría/Tesis/Repo/data/data_out/test_arrays/img_{i}_{df_subset.iloc[i].link}.npy", image)
        return image, value

    def get_train_data(i):
        image, value = get_data(i, df_train, type="train")
        return image, value

    def get_test_data(i):
        image, value = get_data(i, df_test, type="test")
        return image, value

    print()
    print("Benchmarking MSE against the mean")
    print(f"Train MSE: {df_train['var'].var()}")
    print(f"Test MSE: {df_test['var'].var()}")

    print(f"Sample size: {sample}")

    ### Generate Datasets
    ## TRAIN ##
    # Generator for the index
    train_dataset = tf.data.Dataset.from_generator(
        lambda: list(range(df_train.shape[0])),  # The index generator,
        tf.uint32,
    )  # Creates a dataset with only the indexes (0, 1, 2, 3, etc.)

    train_dataset = train_dataset.shuffle(
        buffer_size=int(df_train.shape[0]),
        seed=825,
        reshuffle_each_iteration=True,
    )

    train_dataset = train_dataset.map(
        lambda i: tf.py_function(  # The actual data generator. Passes the index to the function that will process the data.
            func=get_train_data, inp=[i], Tout=[tf.uint8, tf.float32]
        ),
    )

    # FIXME: debería agregar esto...
    # train_dataset = train_dataset.filter(
    #     lambda img, value: (value == 0)
    # )  # Filter out links without image
    train_dataset = train_dataset.batch(32)
    if sample_size > 1:
        train_dataset = train_dataset.repeat(sample_size).prefetch(tf.data.AUTOTUNE)
    else:
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    ## TEST ##
    # Generator for the index
    test_dataset = tf.data.Dataset.from_generator(
        lambda: list(range(df_test.shape[0])),  # The index generator,
        tf.uint8,
    )  # Creates a dataset with only the indexes (0, 1, 2, 3, etc.)

    test_dataset = test_dataset.map(
        lambda i: tf.py_function(  # The actual data generator. Passes the index to the function that will process the data.
            func=get_test_data, inp=[i], Tout=[tf.uint8, tf.float32]
        ),
        num_parallel_calls=1,  # tf.data.experimental.AUTOTUNE,
    )

    test_dataset = test_dataset.batch(128).repeat(10)

    if save_examples == True:
        i = 0
        print("saving train/test examples")
        for x in train_dataset.take(5):
            np.save(
                f"{path_outputs}/{savename}_train_example_{i}_imgs", tfds.as_numpy(x)[0]
            )
            np.save(
                f"{path_outputs}/{savename}_train_example_{i}_labs", tfds.as_numpy(x)[1]
            )
            i += 1
        i = 0
        for x in test_dataset.take(5):
            np.save(
                f"{path_outputs}/{savename}_test_example_{i}_imgs", tfds.as_numpy(x)[0]
            )
            np.save(
                f"{path_outputs}/{savename}_test_example_{i}_labs", tfds.as_numpy(x)[1]
            )
            i += 1

    print()
    print("Dataset generado!")

    return train_dataset, test_dataset


def get_callbacks(
    savename,
    logdir=None,
) -> List[Union[TensorBoard, EarlyStopping, ModelCheckpoint]]:
    """Accepts the model name as a string and returns multiple callbacks for training the keras model.

    Parameters
    ----------
    model_name : str
        The name of the model as a string.

    Returns
    -------
    List[Union[TensorBoard, EarlyStopping, ModelCheckpoint]]
        A list of multiple keras callbacks.
    """

    class CustomLossCallback(tf.keras.callbacks.Callback):
        def __init__(self, log_dir):
            super(CustomLossCallback, self).__init__()
            self.log_dir = log_dir

        def on_epoch_end(self, epoch, logs=None):
            # Save model
            os.makedirs(f"{path_dataout}/models_by_epoch/{savename}", exist_ok=True)
            self.model.save(
                f"{path_dataout}/models_by_epoch/{savename}/{savename}_{epoch}",
                include_optimizer=True,
            )

    tensorboard_callback = TensorBoard(
        log_dir=logdir, histogram_freq=1  # , profile_batch="100,200"
    )
    # use tensorboard --logdir logs/scalars in your command line to startup tensorboard with the correct logs

    # Create an instance of your custom callback
    custom_loss_callback = CustomLossCallback(log_dir=logdir)

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0,  # the training is terminated as soon as the performance measure gets worse from one epoch to the next
        patience=200,  # amount of epochs with no improvements until the model stops
        verbose=2,
        mode="auto",  # the model is stopped when the quantity monitored has stopped decreasing
        restore_best_weights=True,  # restore the best model with the lowest validation error
    )
    # reduce_lr = keras.callbacks.ReduceLROnPlateau(
    #     monitor="val_loss", factor=0.2, patience=10, min_lr=0.0000001
    # )
    model_checkpoint_callback = ModelCheckpoint(
        f"{path_dataout}/models/{savename}",
        monitor="val_loss",
        verbose=1,
        save_best_only=True,  # save the best model
        mode="auto",
        save_freq="epoch",  # save every epoch
    )
    csv_logger = CSVLogger(
        f"{path_dataout}/models_by_epoch/{savename}/{savename}_history.csv", append=True
    )

    return [
        tensorboard_callback,
        # reduce_lr,
        # early_stopping_callback,
        model_checkpoint_callback,
        csv_logger,
        custom_loss_callback,
    ]


def run_model(
    model_function: Model,
    lr: float,
    train_dataset: Iterator,
    test_dataset: Iterator,
    loss: str,
    epochs: int,
    metrics: List[str],
    callbacks: List[Union[TensorBoard, EarlyStopping, ModelCheckpoint]],
    savename: str = "",
):
    """This function runs a keras model with the Ranger optimizer and multiple callbacks. The model is evaluated within
    training through the validation generator and afterwards one final time on the test generator.

    Parameters
    ----------
    model_function : Model
        Keras model function like small_cnn()  or adapt_efficient_net().
    lr : float
        Learning rate.
    train_dataset : Iterator
        tensorflow dataset for the training data.
    test_dataset : Iterator
        tesorflow dataset for the test data.
    loss: str
        Loss function.
    metrics: List[str]
        List of metrics to be used.

    Returns
    -------
    History
        The history of the keras model as a History object. To access it as a Dict, use history.history.
    """
    import tensorflow.python.keras.backend as K

    def get_last_trained_epoch(savename):
        try:
            files = os.listdir(f"{path_dataout}/models_by_epoch/{savename}")
            epochs = [file.split("_")[-1] for file in files]
            epochs = [int(epoch) for epoch in epochs if epoch.isdigit()]
            initial_epoch = max(epochs)
        except:
            os.makedirs(f"{path_dataout}/models_by_epoch/{savename}")
            print("Model not found, running from begining")
            initial_epoch = None

        return initial_epoch

    initial_epoch = get_last_trained_epoch(savename)

    if initial_epoch is None:
        # constructs the model and compiles it
        model = model_function
        model.summary()
        # keras.utils.plot_model(model, to_file=model_name + ".png", show_shapes=True)

        # optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        # opt = tfa.optimizers.RectifiedAdam(learning_rate=lr)
        # # Nesterov Accelerated Gradient (NAG)
        # #   https://kyle-r-kieser.medium.com/tuning-your-keras-sgd-neural-network-optimizer-768536c7ef0
        # optimizer = tf.keras.optimizers.SGD(
        #     learning_rate=lr, momentum=0.9, nesterov=True
        # )  #  1=No friction

        model.compile(optimizer="nadam", loss=loss, metrics=metrics)
        initial_epoch = 0

    else:
        print("Restoring model...")
        model_path = (
            f"{path_dataout}/models_by_epoch/{savename}/{savename}_{initial_epoch}"
        )
        model = keras.models.load_model(model_path)  # load the model from file
        initial_epoch = initial_epoch

    history = model.fit(
        train_dataset,
        epochs=epochs,
        # steps_per_epoch=8545 * sample_size / 32,
        initial_epoch=initial_epoch,
        validation_data=test_dataset,
        callbacks=callbacks,
        workers=2,  # adjust this according to the number of CPU cores of your machine
    )

    # model.evaluate(
    #     test_dataset,
    #     callbacks=callbacks,
    # )
    return model, history  # type: ignore


def plot_predictions_vs_real(df):
    """Genera un scatterplot con la comparación entre los valores reales y los predichos.

    Parameters:
    -----------
    - df: DataFrame con las columnas 'real' y 'pred'. Se recomienda utilizar el test Dataset para validar
    la performance del modelo.

    """

    # Resultado general
    slope, intercept, r, p_value, std_err = scipy.stats.linregress(
        df["real"], df["pred"]
    )

    # Gráfico de correlacion
    sns.set(rc={"figure.figsize": (11.7, 8.27)})
    g = sns.jointplot(
        data=df,
        x="real",
        y="pred",
        kind="reg",
        xlim=(df.real.min(), df.real.max()),
        ylim=(df.real.min(), df.real.max()),
        height=10,
        joint_kws={"line_kws": {"color": "cyan"}},
        scatter_kws={"s": 2},
    )
    g.ax_joint.set_xlabel("Valores Reales", fontweight="bold")
    g.ax_joint.set_ylabel("Valores Predichos", fontweight="bold")

    # Diagonal
    x0, x1 = g.ax_joint.get_xlim()
    y0, y1 = g.ax_joint.get_ylim()
    lims = [max(x0, y0), min(x1, y1)]
    g.ax_joint.plot(lims, lims, "-r", color="orange", linewidth=2)

    # Texto con la regresión
    plt.text(
        0.1,
        0.9,
        f"$y={intercept:.2f}+{slope:.2f}x$; $R^2$={r**2:.2f}",
        transform=g.ax_joint.transAxes,
        fontsize=12,
    )

    return g


def set_model_and_loss_function(
    model_name: str, kind: str, resizing_size: int, weights: str, bands: int = 4
):
    # Diccionario de modelos
    get_model_from_name = {
        "small_cnn": custom_models.small_cnn(resizing_size),  # kind=kind),
        "mobnet_v3_large": custom_models.mobnet_v3_large(
            resizing_size, bands=bands, kind=kind, weights=weights
        ),
        # "resnet152_v2": custom_models.resnet152_v2(kind=kind, weights=weights),
        "effnet_v2_b2": custom_models.effnet_v2_b2(kind=kind, weights=weights),
        "effnet_v2_s": custom_models.effnet_v2_s(kind=kind, weights=weights),
        "effnet_v2_l": custom_models.effnet_v2_l(kind=kind, weights=weights),
        # "effnet_b0": custom_models.effnet_b0(kind=kind, weights=weights),
    }

    # Validación de parámetros
    assert kind in ["reg", "cla"], "kind must be either 'reg' or 'cla'"
    assert (
        model_name in get_model_from_name.keys()
    ), "model_name must be one of the following: " + str(
        list(get_model_from_name.keys())
    )

    # Get model
    model = get_model_from_name[model_name]

    # Set loss and metrics
    if kind == "reg":
        loss = keras.losses.MeanSquaredError()
        # loss = keras.losses.MeanAbsoluteError()
        metrics = [
            # keras.metrics.MeanAbsoluteError(),
            keras.metrics.RootMeanSquaredError(),
            keras.metrics.MeanAbsolutePercentageError(),
            # tfa.metrics.RSquare(),
        ]

    elif kind == "cla":
        loss = keras.losses.CategoricalCrossentropy()
        metrics = [
            keras.metrics.CategoricalAccuracy(),
            keras.metrics.CategoricalCrossentropy(),
        ]

    return model, loss, metrics


def run(
    model_name: str,
    kind: str,
    weights=None,
    image_size=512,
    resizing_size=200,
    tiles=1,
    nbands=4,
    stacked_images=[1],
    sample_size=1,
    small_sample=False,
    n_epochs=100,
    extra="",
):
    """Run all the code of this file.

    Parameters
    ----------
    small_sample : bool, optional
        If you just want to check if the code is working, set small_sample to True, by default False
    """
    log_dir = f"{path_logs}/{model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    if len(stacked_images) > 1:
        savename = f"{model_name}_size{image_size}_tiles{tiles}_sample{sample_size}_stacked{extra}"
    else:
        savename = (
            f"{model_name}_size{image_size}_tiles{tiles}_sample{sample_size}{extra}"
        )

    # ## Set Model & loss function
    # model, loss, metrics = set_model_and_loss_function(
    #     model_name=model_name,
    #     kind=kind,
    #     bands=nbands * len(stacked_images),
    #     resizing_size=resizing_size,
    #     weights=weights,
    # )

    # ### Create train and test dataframes from ICPAG
    df_train, df_test, all_years_datasets = create_train_test_dataframes(
        savename, sat_data="landsat", small_sample=small_sample
    )

    # ## Transform dataframes into datagenerators:
    # #    instead of iterating over census tracts (dataframes), we will generate one (or more) images per census tract
    # print("Setting up data generators...")
    # train_dataset, test_dataset = create_datasets(
    #     df_train=df_train,
    #     df_test=df_test,
    #     all_years_datasets=all_years_datasets,
    #     image_size=image_size,
    #     resizing_size=resizing_size,
    #     nbands=nbands,
    #     stacked_images=stacked_images,
    #     tiles=tiles,
    #     sample=sample_size,
    #     savename=savename,
    #     save_examples=True,
    # )
    # # Get tensorboard callbacks and set the custom test loss computation
    # #   at the end of each epoch
    # callbacks = get_callbacks(
    #     savename=savename,
    #     logdir=log_dir,
    # )

    # # Run model
    # model, history = run_model(
    #     model_function=model,
    #     lr=0.0001, # 0.001 dio cualquier cosa. ## lr=0.00009 para mobnet_v3_20230823-141458
    #     train_dataset=train_dataset,
    #     test_dataset=test_dataset,
    #     loss=loss,
    #     metrics=metrics,
    #     callbacks=callbacks,
    #     epochs=n_epochs,
    #     savename=savename,
    # )

    # Compute metrics
    # hist_df = pd.read_csv(fr"{path_dataout}/models_by_epoch/{savename}/{savename}_history.csv")
    # true_metrics.plot_results(
    #     models_dir=rf"{path_dataout}/models_by_epoch/{savename}",
    #     savename=savename,
    #     tiles=tiles,
    #     size=image_size,
    #     resizing_size=resizing_size,
    #     n_epochs=hist_df.index.max(),
    #     n_bands=nbands,
    #     stacked_images=stacked_images,
    #     generate=True,
    # )

    # Generate gridded predictions & plot examples
    for year in all_years_datasets.keys():
        grid_preds, datasets, extents = grid_predictions.generate_grid(
            savename,
            image_size,
            resizing_size,
            nbands,
            stacked_images,
            year=year,
            generate=True,
        )
        grid_predictions.plot_all_examples(
            datasets, extents, grid_preds, savename, year
        )


if __name__ == "__main__":
    image_size = 32  # FIXME: Creo que solo anda con numeros pares, alguna vez estaría bueno arreglarlo...
    sample_size = 1
    resizing_size = 32
    tiles = 1

    variable = "ln_pred_inc_mean"
    kind = "reg"
    model = "mobnet_v3_large"
    path_repo = r"/mnt/d/Maestría/Tesis/Repo/"
    extra = "_landsat"
    stacked_images = [1]

    # Train the Model
    run(
        model_name=model,
        kind=kind,
        small_sample=False,
        weights=None,
        image_size=image_size,
        sample_size=sample_size,
        resizing_size=resizing_size,
        nbands=10,
        tiles=tiles,
        stacked_images=stacked_images,
        n_epochs=500,
        extra=extra,
    )
