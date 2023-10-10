##############      Configuración      ##############
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from dotenv import dotenv_values

pd.set_option("display.max_columns", None)
try:
    env = dotenv_values("/mnt/d/Maestría/Tesis/Repo/scripts/globals.env")
except:
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

import prediction_tools
import custom_models
import build_dataset
import utils

import sys
import pandas as pd
import os
from typing import Iterator, List, Union, Tuple, Any
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

from tensorflow.keras import layers, models, Model
from tensorflow.keras.callbacks import (
    TensorBoard,
    EarlyStopping,
    ModelCheckpoint,
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


def create_datasets(
    df, kind, image_size, resizing_size, batch_size, save_examples=True
):
    """Accepts four Pandas DataFrames: all your data, the training, validation and test DataFrames. Creates and returns
    keras ImageDataGenerat
    ors. Within this function you can also visualize the augmentations of the ImageDataGenerators.

    Parameters
    ----------
    df : pd.DataFrame
        Your Pandas DataFrame containing all your data.
    train : pd.DataFrame
        Your Pandas DataFrame containing your training data.
    val : pd.DataFrame
        Your Pandas DataFrame containing your validation data.
    test : pd.DataFrame
        Your Pandas DataFrame containing your testing data.

    Returns
    -------
    Tuple[tf.Dataset, tf.Dataset, tf.Dataset, dict]
        A tuple containing the training, validation and test datasets, and a dictionary with each dataset's filepaths.
    """

    # Train, validation and test split
    train = df[df.type == "train"]
    test = df[df.type == "test"]
    # test, val = train_test_split(test, test_size=0.5, shuffle=True, random_state=528)

    assert pd.concat([train, test]).sort_index().equals(df)

    ### Benchmarking MSE against the mean
    print("Benchmarking MSE against the mean")
    print("Train MSE: ", train["var"].var())
    print("Test MSE: ", test["var"].var())

    ### dataframes to tf.data.Dataset

    def tf_process_image(file_path, label):
        img = np.load(file_path)
        img = utils.process_image(img, resizing_size)
        img = tf.convert_to_tensor(img, dtype=tf.float32)
        label = tf.cast(label, tf.float32)

        return img, label

    # Create tf.data pipeline for each dataset
    datasets = []
    filenames_l = []
    for dataframe in [train, test]:
        # Get list of filenames and corresponding list of labels
        filenames = tf.constant(dataframe["image"].to_list())
        if kind == "reg":
            labels = tf.constant(dataframe["var"].to_list())
        elif kind == "cla":
            labels = tf.one_hot(
                (df["var"] - 1).to_list(), 10
            )  # Fist class corresponds to decile 1, etc.

        # Create a dataset from the filenames and labels
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        dataset = dataset.map(
            lambda file, label: tf.numpy_function(
                tf_process_image, [file, label], (tf.float32, tf.float32)
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )  # Parse every image in the dataset using `map`
        datasets += [dataset]

        # Store filenames for later use
        filenames_l += [dataframe["image"].to_list()]

    train_dataset, test_dataset = datasets
    filenames = {
        "train": filenames_l[0],
        "test": filenames_l[1],
    }  # , "val": filenames_l[2]}

    # Create a generator.
    rng = tf.random.Generator.from_seed(123, alg="philox")

    def augment(image_label, seed):
        image, label = image_label
        # Adjust brightness
        image = tf.image.stateless_random_brightness(image, max_delta=0.2, seed=seed)
        image = tf.clip_by_value(image, 0, 1)
        # Flip images # FIXME: ¿esta bien hacer esto por batches o tendría que cambiar el seed acá
        tf.image.stateless_random_flip_left_right(image, seed)
        tf.image.stateless_random_flip_up_down(image, seed)

        return image, label

        # Split tensor
        # image_rgb = image
        # [:, :, :3]
        # image_nr = image[:, :, 3:4]

        # RGB augmentation
        # image_rgb = tf.image.random_flip_left_right(image_rgb)
        # image_rgb = tf.image.random_flip_up_down(image_rgb)
        # image_rgb = tf.image.random_contrast(image_rgb, lower=0.2, upper=1.5)
        # image_rgb = tf.image.random_brightness(image_rgb, max_delta=0.2)
        # image_rgb = tf.image.resize(
        #     image_rgb, (resizing_size, resizing_size), antialias=True
        # )

        # # Near infrared augmentation
        # image_rgb = tf.image.resize(
        #     image_rgb, (resizing_size, resizing_size), antialias=True
        # )

        # Rebuild image channels
        # test_image = tf.concat([image_rgb, image_nr], 2).numpy()

        # return image_rgb

    # Create a wrapper function for updating seeds.
    def data_augmentation(x, y):
        seed = rng.make_seeds(2)[0]
        image, label = augment((x, y), seed)
        return image, label

    # Prepare dataset for training
    train_dataset = (
        train_dataset.shuffle(1000).batch(batch_size)
        # .map(
        #     data_augmentation,
        #     num_parallel_calls=tf.data.AUTOTUNE,
        # )
        .prefetch(tf.data.AUTOTUNE)
    )
    test_dataset = test_dataset.batch(batch_size)
    # val_dataset = val_dataset.batch(batch_size)

    if save_examples == True:
        i = 0
        for x in train_dataset.take(5):
            np.save(f"{path_outputs}/train_example_{i}_imgs", tfds.as_numpy(x)[0])
            np.save(f"{path_outputs}/train_example_{i}_labs", tfds.as_numpy(x)[1])
            i += 1
        i = 0
        for x in test_dataset.take(5):
            np.save(f"{path_outputs}/test_example_{i}_imgs", tfds.as_numpy(x)[0])
            np.save(f"{path_outputs}/test_example_{i}_labs", tfds.as_numpy(x)[1])
            i += 1
    print("Dataset generado!")
    return train_dataset, test_dataset, filenames


def get_batch_predictions(model, batch_images):
    """Computa las predicciones del batch

    Parameters:
    -----------
    model: tf.keras.Model, modelo entrenado
    images: np.array con las imágenes a predecir (batch_size, img_size, img_size, bands)
    batch_real_value: np.array con el valor real del radio censal correspondiente (batch_size)
    metric: function, función que computa el error de predicción. Por defecto es np.mean
    """
    to_predict = tf.data.Dataset.from_tensor_slices(batch_images)

    to_predict = to_predict.batch(400)
    predictions = model.predict(to_predict)
    predictions = predictions.flatten()

    return predictions


def compute_custom_loss(
    model,
    metadata,
    tiles,
    size,
    resizing_size,
    bias,
    sample,
    to8bit,
    verbose=False,
    trim_size=True,
):
    """
    Calcula el ECM del conjunto de predicción.

    Carga las bases necesarias, itera sobre los radios censales, genera las imágenes
    en forma de grilla y compara las predicciones de esas imágenes con el valor real
    del radio censal.

    Parameters:
    -----------
    metadata: pd.DataFrame, dataframe con los metadatos de las imágenes
    tiles: int, cantidad de imágenes a generar por lado
    size: int, tamaño de la imagen a generar, en píxeles
    resizing_size: int, tamaño al que se redimensiona la imagen
    bias: int, cantidad de píxeles que se mueve el punto aleatorio de las tiles
    to8bit: bool, si es True, convierte la imagen a 8 bits

    Returns:
    --------
    mse: float, error cuadrático medio del conjunto de predicción

    """
    import random

    if verbose == False:
        blockPrint()

    # Inicializo arrays
    batch_images = np.empty((0, resizing_size, resizing_size, 4))
    batch_link_names = np.empty((0))
    batch_real_values = np.empty((0))
    batch_predictions = np.empty((0))
    all_link_names = np.empty((0))
    all_predictions = np.empty((0))
    all_real_values = np.empty((0))

    # Cargar bases de datos
    datasets, extents = build_dataset.load_satellite_datasets()
    icpag = build_dataset.load_icpag_dataset()
    icpag = build_dataset.assign_links_to_datasets(icpag, extents, verbose=False)

    # Filtro Radios demasiado grandes (tardan horas en generar la cuadrícula y es puro campo...)
    if trim_size:
        icpag = icpag[icpag["AREA"] <= 200000]  # Remove rc that are too big
    links = metadata["link"].astype(str).str.zfill(9).unique()
    links = [link for link in links if link in icpag.link.unique()]
    # links = random.sample(links, 30)
    len_links = len(links)

    # Creo la carpeta de test
    test_folder = rf"{path_imgs}/test_size{size}_tiles{tiles}_sample{sample}"
    os.makedirs(test_folder, exist_ok=True)
    print(len_links)

    # Loop por radio censal. Si está la imagen la usa, sino la genera.
    for n, link in enumerate(links):
        print(f"{link}: {n}/{len_links}")
        # Abre/genera la imagen
        file = rf"{test_folder}/test_{link}.npy"
        if os.path.isfile(file):
            images = np.load(file)
        else:
            print("Generando...")
            link_dataset = build_dataset.get_dataset_for_link(icpag, datasets, link)
            print("listo")
            if tiles==1:
                images, points, bounds = build_dataset.get_gridded_images_for_link(
                    link_dataset,
                    icpag,
                    link,
                    tiles,
                    size,
                    resizing_size,
                    bias,
                    sample,
                    to8bit,
                )
            else: # If tiles >2 then the dataset is too big
                images, points, bounds = build_dataset.get_random_images_for_link(
                    link_dataset, 
                    icpag, 
                    link, 
                    tiles, 
                    size,
                    resizing_size, 
                    bias, 
                    sample, 
                    to8bit
                )
            print("imagen generada")
            if len(images) == 0:
                # No images where returned from this census tract, so no error to compute...
                continue
            images = np.array(images)
            np.save(file, images)

        # Obtener el error de estimación del radio censal
        link_real_value = metadata.loc[metadata["link"] == int(link), "var"].values[0]

        # Agrega al batch de valores reales / imagenes para la prediccion
        q_images = images.shape[0]
        link_names = np.array([link] * q_images)
        real_values = np.array([link_real_value] * q_images)

        batch_images = np.concatenate([images, batch_images], axis=0)
        batch_link_names = np.concatenate([link_names, batch_link_names], axis=0)
        batch_real_values = np.concatenate([real_values, batch_real_values], axis=0)

        if batch_real_values.shape[0] > 128:
            batch_predictions = get_batch_predictions(model, batch_images)

            # Store data
            all_link_names = np.concatenate([all_link_names, batch_link_names])
            all_predictions = np.concatenate([all_predictions, batch_predictions])
            all_real_values = np.concatenate([all_real_values, batch_real_values])

            # Restore batches to empty
            batch_images = np.empty((0, resizing_size, resizing_size, 4))
            batch_link_names = np.empty((0))
            batch_real_values = np.empty((0))
            batch_predictions = np.empty((0))

    if verbose == False:
        enablePrint()

    # Creo dataframe para exportar:
    d = {
        "link": all_link_names,
        "predictions": all_predictions,
        "real_value": all_real_values,
    }
    df_preds = pd.DataFrame(data=d)

    df_preds["mean_prediction"] = df_preds.groupby(by="link").predictions.transform(
        "mean"
    )
    df_preds["error"] = df_preds["mean_prediction"] - df_preds["real_value"]
    df_preds["sq_error"] = df_preds["error"] ** 2
    mse = df_preds.drop_duplicates(subset=["link"]).sq_error.mean()

    return df_preds, mse


def compute_predictions_dataset(
    model,
    metadata,
    tiles,
    size,
    resizing_size,
    bias,
    sample,
    to8bit,
    verbose=False,
    trim_size=True,
):
    """
    Calcula predicciones y errores para un dataset determinado.

    Carga las bases necesarias, itera sobre los radios censales, genera las imágenes
    en forma de grilla y compara las predicciones de esas imágenes con el valor real
    del radio censal.

    Parameters:
    -----------
    metadata: pd.DataFrame, dataframe con los metadatos de las imágenes
    tiles: int, cantidad de imágenes a generar por lado
    size: int, tamaño de la imagen a generar, en píxeles
    resizing_size: int, tamaño al que se redimensiona la imagen
    bias: int, cantidad de píxeles que se mueve el punto aleatorio de las tiles
    to8bit: bool, si es True, convierte la imagen a 8 bits

    Returns:
    --------
    mse: float, error cuadrático medio del conjunto de predicción

    """
    import random

    if verbose == False:
        blockPrint()

    # Inicializo arrays
    batch_images = np.empty((0, resizing_size, resizing_size, 4))
    batch_link_names = np.empty((0))
    batch_real_values = np.empty((0))
    batch_predictions = np.empty((0))
    all_link_names = np.empty((0))
    all_predictions = np.empty((0))
    all_real_values = np.empty((0))

    # Cargar bases de datos
    datasets, extents = build_dataset.load_satellite_datasets()
    icpag = build_dataset.load_icpag_dataset()
    icpag = build_dataset.assign_links_to_datasets(icpag, extents, verbose=False)

    # Filtro Radios demasiado grandes (tardan horas en generar la cuadrícula y es puro campo...)
    if trim_size:
        icpag = icpag[icpag["AREA"] <= 4000000]  # Remove rc that are too big
    links = metadata["link"].astype(str).str.zfill(9).unique()
    links = [link for link in links if link in icpag.link.unique()]
    # links = random.sample(links, 30)
    len_links = len(links)

    # Creo la carpeta de test
    test_folder = rf"{path_dataout}/test_size{size}_tiles{tiles}_sample{sample}"
    os.makedirs(test_folder, exist_ok=True)

    # Loop por radio censal. Si está la imagen la usa, sino la genera.
    for n, link in enumerate(links):
        print(f"{link}: {n}/{len_links}")
        # Abre/genera la imagen
        file = rf"{test_folder}/test_{link}.npy"
        if os.path.isfile(file):
            images = np.load(file)
        else:
            link_dataset = build_dataset.get_dataset_for_link(icpag, datasets, link)
            images, points, bounds = build_dataset.get_gridded_images_for_link(
                link_dataset,
                icpag,
                link,
                tiles,
                size,
                resizing_size,
                bias,
                sample,
                to8bit,
            )
            if len(images) == 0:
                # No images where returned from this census tract, so no error to compute...
                continue
            images = np.array(images)
            np.save(file, images)

        # Obtener el error de estimación del radio censal
        link_real_value = metadata.loc[metadata["link"] == int(link), "var"].values[0]

        # Agrega al batch de valores reales / imagenes para la prediccion
        q_images = images.shape[0]
        link_names = np.array([link] * q_images)
        real_values = np.array([link_real_value] * q_images)

        batch_images = np.concatenate([images, batch_images], axis=0)
        batch_link_names = np.concatenate([link_names, batch_link_names], axis=0)
        batch_real_values = np.concatenate([real_values, batch_real_values], axis=0)

        if batch_real_values.shape[0] > 128:
            batch_predictions = get_batch_predictions(model, batch_images)

            # Store data
            all_link_names = np.concatenate([all_link_names, batch_link_names])
            all_predictions = np.concatenate([all_predictions, batch_predictions])
            all_real_values = np.concatenate([all_real_values, batch_real_values])

            # Restore batches to empty
            batch_images = np.empty((0, resizing_size, resizing_size, 4))
            batch_link_names = np.empty((0))
            batch_real_values = np.empty((0))
            batch_predictions = np.empty((0))

    if verbose == False:
        enablePrint()

    # Creo dataframe para exportar:
    d = {
        "link": all_link_names,
        "predictions": all_predictions,
        "real_value": all_real_values,
    }
    df_preds = pd.DataFrame(data=d)

    df_preds["mean_prediction"] = df_preds.groupby(by="link").predictions.transform(
        "mean"
    )
    df_preds["error"] = df_preds["mean_prediction"] - df_preds["real_value"]
    df_preds["sq_error"] = df_preds["error"] ** 2
    mse = df_preds.drop_duplicates(subset=["link"]).sq_error.mean()

    return df_preds, mse


def compute_custom_loss_all_epochs(
    log_dir,
    models_dir,
    model_name,
    metadata,
    tiles,
    size,
    resizing_size,
    bias,
    sample,
    to8bit,
    n_epochs=20,
):
    """
    Calcula el ECM del conjunto de predicción.

    Carga las bases necesarias, itera sobre los radios censales, genera las imágenes
    en forma de grilla y compara las predicciones de esas imágenes con el valor real
    del radio censal.

    Parameters:
    -----------
    metadata: pd.DataFrame, dataframe con los metadatos de las imágenes
    tiles: int, cantidad de imágenes a generar por lado
    size: int, tamaño de la imagen a generar, en píxeles
    resizing_size: int, tamaño al que se redimensiona la imagen
    bias: int, cantidad de píxeles que se mueve el punto aleatorio de las tiles
    to8bit: bool, si es True, convierte la imagen a 8 bits

    Returns:
    --------
    mse: float, error cuadrático medio del conjunto de predicción

    """
    # blockPrint()
    mse_epochs = {epoch: 0 for epoch in range(n_epochs)}
    models = {
        epoch: tf.keras.models.load_model(
            f"{models_dir}/{model_name}_{epoch}", compile=True
        )
        for epoch in range(n_epochs)
    }
    datasets, extents = build_dataset.load_satellite_datasets()
    icpag = build_dataset.load_icpag_dataset()
    icpag = build_dataset.assign_links_to_datasets(icpag, extents, verbose=False)
    links = (
        metadata.loc[metadata.type == "test", "link"].astype(str).str.zfill(9).unique()
    )

    len_links = len(links)

    test_folder = rf"{path_dataout}/test_size{size}_tiles{tiles}_sample{sample}"
    os.makedirs(test_folder, exist_ok=True)

    for n, link in enumerate(links):
        try:
            file = rf"{test_folder}/test_{link}.npy"
            if os.path.isfile(file):
                images = np.load(file)
            else:
                link_dataset = build_dataset.get_dataset_for_link(icpag, datasets, link)
                images, points, bounds = build_dataset.get_gridded_images_for_link(
                    link_dataset,
                    icpag,
                    link,
                    tiles,
                    size,
                    resizing_size,
                    bias,
                    sample,
                    to8bit,
                )
                images = np.array(images)
                np.save(file, images)

            # Obtener el error de estimación del radio censal
            link_real_value = metadata.loc[metadata["link"] == int(link), "var"].values[
                0
            ]
            for epoch in range(1, n_epochs):
                link_error = get_link_error(
                    models[epoch], images, resizing_size, link_real_value
                )

                # Actualizar el error cuadratico medio
                #   - Suma ponderada del error anterior y el nuevo error
                mse_previo = mse_epochs[epoch]
                mse_updated = (mse_previo * n + link_error**2) / (n + 1)
                mse_epochs[epoch] = mse_updated

        except:
            print(f"problema con link {link}...")
            continue

    # enablePrint()

    for epoch in range(0, n_epochs):
        custom_loss = mse_epochs[epoch]
        print(f"Epoch {epoch}/{n_epochs}: True Mean Squared Error: {custom_loss}")

        # Log the custom loss to TensorBoard
        with tf.summary.create_file_writer(log_dir).as_default():
            tf.summary.scalar("true_mean_squared_error", custom_loss, step=epoch)


def get_callbacks(
    model_name: str,
    loss: str,
    metadata,
    tiles,
    size,
    resizing_size,
    bias=2,
    test_sample=1,
    to8bit=True,
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
            print("Calculando el verdadero ECM. Esto puede tardar un rato...")
            # Calculate your custom loss here (replace this with your actual custom loss calculation)
            df_prediciones, mse = compute_custom_loss(
                self.model,
                metadata[metadata.type == "test"],
                tiles,
                size,
                resizing_size,
                bias,
                test_sample,
                to8bit,
                verbose=False,
            )
            print(f"True Mean Squared Error: {mse}")

            # Log the custom loss to TensorBoard
            with tf.summary.create_file_writer(self.log_dir).as_default():
                tf.summary.scalar("true_mean_squared_error", mse, step=epoch)

            # Save model
            savename = f"{model_name}_size{size}_tiles{tiles}_sample{test_sample}"
            os.makedirs(f"{path_dataout}/models_by_epoch/{savename}", exist_ok=True)
            self.model.save(
                f"{path_dataout}/models_by_epoch/{savename}/{savename}_{epoch}",
                include_optimizer=True,
            )
            # Save predictions
            df_prediciones.to_csv(
                f"{path_dataout}/models_by_epoch/{savename}/{savename}_{epoch}.csv"
            )

    tensorboard_callback = TensorBoard(
        log_dir=logdir, histogram_freq=1, profile_batch="500,520"
    )
    # use tensorboard --logdir logs/scalars in your command line to startup tensorboard with the correct logs

    # Create an instance of your custom callback
    custom_loss_callback = CustomLossCallback(log_dir=logdir)

    # FIXME: Sacar esto
    # early_stopping_callback = EarlyStopping(
    #     monitor="val_loss",
    #     min_delta=0,  # the training is terminated as soon as the performance measure gets worse from one epoch to the next
    #     patience=30,  # amount of epochs with no improvements until the model stops
    #     verbose=2,
    #     mode="auto",  # the model is stopped when the quantity monitored has stopped decreasing
    #     restore_best_weights=True,  # restore the best model with the lowest validation error
    # )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=10, min_lr=0.0000001
    )
    savename = f"{model_name}_size{size}_tiles{tiles}_sample{test_sample}"
    model_checkpoint_callback = ModelCheckpoint(
        f"{path_dataout}/models/{savename}",
        monitor="val_loss",
        verbose=1,
        save_best_only=True,  # save the best model
        mode="auto",
        save_freq="epoch",  # save every epoch
    )  # saving eff_net takes quite a bit of time

    return [
        tensorboard_callback,
        # reduce_lr,
        # early_stopping_callback,
        model_checkpoint_callback,
        custom_loss_callback,
    ]


def run_model(
    model_name: str,
    model_function: Model,
    lr: float,
    train_dataset: Iterator,
    test_dataset: Iterator,
    loss: str,
    metrics: List[str],
    callbacks: List[Union[TensorBoard, EarlyStopping, ModelCheckpoint]],
    model_path: str = None,
    epochs: int = 20,
    initial_epoch: int = 0,
):
    """This function runs a keras model with the Ranger optimizer and multiple callbacks. The model is evaluated within
    training through the validation generator and afterwards one final time on the test generator.

    Parameters
    ----------
    model_name : str
        The name of the model as a string.
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
    model_path : str, optional
        Path to the model if restored from previous training, by default None.
    epochs : int, optional
        Number of epochs to train, by default 20.
    initial_epoch : int, optional
        Initial epoch, by default 0. If restoring training


    Returns
    -------
    History
        The history of the keras model as a History object. To access it as a Dict, use history.history.
    """
    import tensorflow.python.keras.backend as K

    if model_path is None:
        # constructs the model and compiles it
        model = model_function
        model.summary()
        # keras.utils.plot_model(model, to_file=model_name + ".png", show_shapes=True)

        optimizer = tf.keras.optimizers.Adam()
        # opt = tfa.optimizers.RectifiedAdam(learning_rate=lr)
        # # Nesterov Accelerated Gradient (NAG)
        # #   https://kyle-r-kieser.medium.com/tuning-your-keras-sgd-neural-network-optimizer-768536c7ef0
        # optimizer = tf.keras.optimizers.SGD(
        #     learning_rate=lr, momentum=0.9, nesterov=True
        # )  #  1=No friction

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        initial_epoch = 0
    else:
        # assert os.path.isfolder(
        #     model_path
        # ), "model_path must be a valid path to a model"
        model = keras.models.load_model(model_path)  # load the model from file
        initial_epoch = initial_epoch

    history = model.fit(
        train_dataset,
        epochs=epochs,
        initial_epoch=initial_epoch,
        validation_data=test_dataset,
        callbacks=callbacks,
        workers=2,  # adjust this according to the number of CPU cores of your machine
    )

    model.evaluate(
        test_dataset,
        callbacks=callbacks,
    )
    return model, history  # type: ignore


def plot_predictions_vs_real(df):
    """Genera un scatterplot con la comparación entre los valores reales y los predichos.

    Parameters:
    -----------
    - df: DataFrame con las columnas 'real' y 'pred'. Se recomienda utilizar el test Dataset para validar
    la performance del modelo.

    """
    import scipy

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
    model_name: str, kind: str, image_size: int, resizing_size: int, weights: str
):
    # Diccionario de modelos
    get_model_from_name = {
        "small_cnn": custom_models.small_cnn(resizing_size),  # kind=kind),
        "mobnet_v3": custom_models.mobnet_v3(resizing_size, kind=kind, weights=weights),
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
        metrics = [
            keras.metrics.MeanAbsoluteError(),
            keras.metrics.MeanSquaredError(),
            # keras.metrics.R2Score(),
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
    pred_variable: str,
    kind: str,
    weights=None,
    image_size=512,
    resizing_size=200,
    tiles=1,
    sample_size=10,
    small_sample=False,
    n_epochs=100,
    initial_epoch=0,
    model_path=None,
):
    """Run all the code of this file.

    Parameters
    ----------
    small_sample : bool, optional
        If you just want to check if the code is working, set small_sample to True, by default False
    """
    log_dir = f"{path_logs}/{model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    ### Set Model & loss function
    model, loss, metrics = set_model_and_loss_function(
        model_name=model_name,
        kind=kind,
        image_size=image_size,
        resizing_size=resizing_size,
        weights=weights,
    )

    print("Reading dataset...")
    # Open dataframe with files and labels
    df = pd.read_csv(
        rf"{path_imgs}/train_size{image_size}_tiles{tiles}_sample{sample_size}/metadata.csv"
    )
    print("Dataset loaded!")

    print("Cleaning dataset...")
    # Clean dataframe and create datasets
    # df = df[df["sample"] <= 2]
    df = df.dropna(how="any").reset_index(drop=True)
    if small_sample:
        df = df.sample(2000, random_state=825).reset_index(drop=True)

    # assert all([os.path.isfile(img) for img in df.image.values])

    print("Moving dataset to tensorflow ds...")
    train_dataset, test_dataset, filenames = create_datasets(
        df=df,
        kind=kind,
        image_size=image_size,
        resizing_size=resizing_size,
        batch_size=32,
    )
    val_dataset = [0]

    # Run model
    callbacks = get_callbacks(
        model_name,
        loss,
        metadata=df,
        tiles=tiles,
        size=image_size,
        resizing_size=resizing_size,
        test_sample=20,
        logdir=log_dir,
    )

    model, history = run_model(
        model_name=model_name,
        model_function=model,
        lr=0.0001,  # lr=0.00009 para mobnet_v3_20230823-141458
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        loss=loss,
        metrics=metrics,
        callbacks=callbacks,
        epochs=n_epochs,
        initial_epoch=initial_epoch,
        model_path=model_path,
    )

    predicted_array = prediction_tools.get_predicted_values(
        model, test_dataset, kind="reg"
    )
    real_array = prediction_tools.get_real_values(test_dataset)

    # Creo dataframe para exportar:
    d = {
        "filename": filenames["test"],
        "real": real_array,
        "pred": predicted_array.flatten(),
    }
    df_prediciones = pd.DataFrame(data=d)
    print("R2 matrix:", df_prediciones[["real", "pred"]].corr() ** 2)
    df_prediciones.to_parquet(
        f"{path_dataout}/preds_{pred_variable}_{model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.parquet"
    )

    # Genero gráficos de predicciones en el set de prueba
    g = plot_predictions_vs_real(df_prediciones)
    g.savefig(
        rf"{path_dataout}/scatterplot_{pred_variable}_{model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.png",
        dpi=300,
    )

    # # Evaluate model in test dataset
    # compute_custom_loss_all_epochs(
    #     log_dir=log_dir,
    #     models_dir=f"{path_dataout}/models_by_epoch",
    #     model_name=model_name,
    #     metadata=df,
    #     size=image_size,
    #     resizing_size=resizing_size,
    #     tiles=1,
    #     bias=2,
    #     sample=1,
    #     to8bit=True,
    #     n_epochs=n_epochs,
    # )


if __name__ == "__main__":
    # import argparse

    variable_models = {
        "rmax": "reg",
        "pred_inc_mean": "reg",
        "pred_inc_p50": "reg",
        # "rmax_c": "reg",
        "nbi_rc_val": "reg",
        "icv2010": "reg",
        "pm2": "reg",
        "viv_part": "reg",
        # "rmax_d": "cla",
        # "icv2010_d": "cla",
    }

    image_size = 200
    sample_size = 1
    # parser = argparse.ArgumentParser(description="Model setup")
    # parser.add_argument(
    #     "--model",
    #     dest="model",
    #     type=str,
    #     nargs="?",
    #     default="mobnet_v3",
    #     help="Name of the model",
    # )
    # parser.add_argument(
    #     "--var",
    #     dest="var",
    #     type=str,
    #     nargs="?",
    #     default="all",
    #     help="Variable to run the model",
    # )

    # args = parser.parse_args()
    # print(args.model)
    # print(args.var)

    # model = args.model
    # vars = [args.var]

    model = "small_cnn"
    vars = ["pred_inc_mean"]
    if (vars == ["all"]) or (vars == [None]):
        vars = variable_models.keys()
    print(vars)
    for var in vars:
        kind = variable_models[var]

        print("#######################################################")
        print(f"Running model for {var} ({kind})")
        print("#######################################################")
        run(
            model_name=model,
            pred_variable=var,
            kind=kind,
            image_size=image_size,
            sample_size=sample_size,
            small_sample=True,
            tiles=tiles,
            weights="imagenet",
        )
