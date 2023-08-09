##############      Configuración      ##############
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from dotenv import dotenv_values

pd.set_option("display.max_columns", None)
env = dotenv_values("/mnt/d/Maestría/Tesis/Repo/scripts/globals.env")

path_proyecto = env["PATH_PROYECTO"]
path_datain = env["PATH_DATAIN"]
path_dataout = env["PATH_DATAOUT"]
path_scripts = env["PATH_SCRIPTS"]
path_satelites = env["PATH_SATELITES"]
path_logs = env["PATH_LOGS"]
path_outputs = env["PATH_OUTPUTS"]
# path_programas  = globales[7]
###############################################

import prediction_tools
import custom_models

import sys
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from typing import Iterator, List, Union, Tuple, Any
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

from tensorflow.keras import layers, models, Model
from tensorflow.keras.callbacks import (
    TensorBoard,
    EarlyStopping,
    ModelCheckpoint,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import (
    MobileNetV3Small,
    EfficientNetB0,
    EfficientNetV2B0,
)

# the next 3 lines of code are for my machine and setup due to https://github.com/tensorflow/tensorflow/issues/43174

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def create_datasets(df, kind, image_size, sample_size, small_sample=False):
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

    from sklearn.model_selection import train_test_split

    # FIXME: hacer set de test arbitrario
    # Train, validation and test split
    train, test = train_test_split(df, test_size=0.2, shuffle=True, random_state=825)
    test, val = train_test_split(test, test_size=0.5, shuffle=True, random_state=528)
    len_test = len(test)

    assert pd.concat([train, test, val]).sort_index().equals(df)

    ### Benchmarking MSE against the mean
    print("Benchmarking MSE against the mean")
    print("Train MSE: ", test["var"].var())

    ### dataframes to tf.data.Dataset

    def process_image(file_path, label):
        img = np.load(file_path)
        img = np.moveaxis(
            img, 0, 2
        )  # Move axis so the original [4, 512, 512] becames [512, 512, 4]
        img = tf.convert_to_tensor(img / 255, dtype=tf.float32)
        label = tf.cast(label, tf.float32)

        return img, label

    # Create tf.data pipeline for each dataset
    datasets = []
    filenames_l = []
    for dataframe in [train, test, val]:
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
                process_image, [file, label], (tf.float32, tf.float32)
            )
        )  # Parse every image in the dataset using `map`
        datasets += [dataset]

        # Store filenames for later use
        filenames_l += [dataframe["image"].to_list()]

    train_dataset, test_dataset, val_dataset = datasets
    filenames = {"train": filenames_l[0], "test": filenames_l[1], "val": filenames_l[2]}

    ### augmentations, batching and prefetching
    # Create a data augmentation stage with horizontal flipping, rotations, zooms
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip(
                "horizontal_and_vertical",
                seed=825,
                input_shape=(image_size, image_size, 4),
            ),
            layers.RandomRotation(0.3, fill_mode="reflect", seed=825),
            layers.RandomTranslation(0.3, 0.3, fill_mode="reflect", seed=825),
            layers.RandomHeight(0.3),
            layers.RandomWidth(0.3),
            layers.RandomZoom(0.3, seed=825),
            layers.RandomContrast(0.3, seed=825),
            layers.RandomBrightness(0.4, seed=825),
            # layers.RandomCrop(image_size, image_size, seed=825),
            # layers.Resizing(image_size, image_size),
        ],
        name="data_augmentation",
    )

    # Prepare dataset for training
    train_dataset = (
        train_dataset.shuffle(round(len(filenames_l[0]) / 10))
        .batch(64)
        .map(lambda x, y: (data_augmentation(x), y))
        .prefetch(tf.data.AUTOTUNE)
    )
    test_dataset = test_dataset.batch(64)
    val_dataset = val_dataset.batch(64)

    return train_dataset, test_dataset, val_dataset, filenames


def get_callbacks(
    model_name: str, loss: str
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

    logdir = (
        "R:/Tesis Nico/Códigos/logs/scalars/"
        + model_name
        + "_"
        + datetime.now().strftime("%Y%m%d-%H%M%S")
    )  # create a folder for each model.

    tensorboard_callback = TensorBoard(
        log_dir=logdir, histogram_freq=1, profile_batch="500,520"
    )
    # use tensorboard --logdir logs/scalars in your command line to startup tensorboard with the correct logs

    # FIXME: Sacar esto
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0,  # the training is terminated as soon as the performance measure gets worse from one epoch to the next
        patience=30,  # amount of epochs with no improvements until the model stops
        verbose=2,
        mode="auto",  # the model is stopped when the quantity monitored has stopped decreasing
        restore_best_weights=True,  # restore the best model with the lowest validation error
    )

    model_checkpoint_callback = ModelCheckpoint(
        f"{path_dataout}/models/" + model_name + ".h5",
        monitor="val_loss",
        verbose=0,
        save_best_only=True,  # save the best model
        mode="auto",
        save_freq="epoch",  # save every epoch
    )  # saving eff_net takes quite a bit of time

    return [
        tensorboard_callback,
        early_stopping_callback,
        model_checkpoint_callback,
    ]


def run_model(
    model_name: str,
    model_function: Model,
    lr: float,
    train_generator: Iterator,
    validation_generator: Iterator,
    test_generator: Iterator,
    loss: str,
    metrics: List[str],
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
    train_generator : Iterator
        keras ImageDataGenerators for the training data.
    validation_generator : Iterator
        keras ImageDataGenerators for the validation data.
    test_generator : Iterator
        keras ImageDataGenerators for the test data.

    Returns
    -------
    History
        The history of the keras model as a History object. To access it as a Dict, use history.history.
    """

    callbacks = get_callbacks(model_name, loss)
    model = model_function
    model.summary()
    # keras.utils.plot_model(model, to_file=model_name + ".png", show_shapes=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    # opt = tfa.optimizers.RectifiedAdam(learning_rate=lr)
    # # Nesterov Accelerated Gradient (NAG)
    # #   https://kyle-r-kieser.medium.com/tuning-your-keras-sgd-neural-network-optimizer-768536c7ef0
    # opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)  #  1=No friction
    # optimizer = tfa.optimizers.Lookahead(opt, sync_period=6, slow_step_size=0.5)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    history = model.fit(
        train_generator,
        epochs=50,
        validation_data=validation_generator,
        shuffle=True,
        callbacks=callbacks,
        workers=2,  # adjust this according to the number of CPU cores of your machine
    )

    model.evaluate(
        test_generator,
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
        xlim=(0, df.real.max()),
        ylim=(0, df.real.max()),
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


def run(
    model_name: str,
    pred_variable: str,
    kind: str,
    weights=None,
    image_size=512,
    sample_size=10,
    small_sample=False,
):
    """Run all the code of this file.

    Parameters
    ----------
    small_sample : bool, optional
        If you just want to check if the code is working, set small_sample to True, by default False
    """
    # Diccionario de modelos
    get_model_from_name = {
        "small_cnn": custom_models.small_cnn(image_size),  # kind=kind),
        "mobnet_v3": custom_models.mobnet_v3(kind=kind, weights=weights),
        # "resnet152_v2": custom_models.resnet152_v2(kind=kind, weights=weights),
        "effnet_v2_b0": custom_models.effnet_v2_b0(kind=kind, weights=weights),
        "effnet_v2_s": custom_models.effnet_v2_s(kind=kind, weights=weights),
        "effnet_v2_l": custom_models.effnet_v2_l(kind=kind, weights=weights),
        # "effnet_b0": custom_models.effnet_b0(kind=kind, weights=weights),
    }

    # Open dataframe with files and labels
    df = pd.read_csv(
        rf"{path_dataout}/size{image_size}_sample{sample_size}/metadata.csv"
    )

    # Validación de parámetros
    assert kind in ["reg", "cla"], "kind must be either 'reg' or 'cla'"
    assert (
        model_name in get_model_from_name.keys()
    ), "model_name must be one of the following: " + str(
        list(get_model_from_name.keys())
    )
    # assert (
    #     pred_variable in df.columns
    # ), "pred_variable must be one of the following: " + str(list(df.columns))
    # assert "image" in df.columns, 'df must have a column called "image"'

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

    # Clean dataframe and create datasets
    df = df.dropna(how="any").reset_index(drop=True)
    if small_sample:
        df = df.sample(500, random_state=825).reset_index(drop=True)
    assert all([os.path.isfile(img) for img in df.image.values])

    train_dataset, test_dataset, val_dataset, filenames = create_datasets(
        df=df,
        kind=kind,
        small_sample=small_sample,
        image_size=image_size,
        sample_size=sample_size,
    )

    # Run model
    model, history = run_model(
        model_name=model_name,
        model_function=get_model_from_name[model_name],
        lr=0.001,  # lr=0.00005 para v0
        train_generator=train_dataset,
        validation_generator=val_dataset,
        test_generator=test_dataset,
        loss=loss,
        metrics=metrics,
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
    print("Correlation matrix:", df_prediciones[["real", "pred"]].corr())
    df_prediciones.to_parquet(
        f"{path_dataout}/preds_{model_name}_{pred_variable}.parquet"
    )

    # Genero gráficos de predicciones en el set de prueba
    g = plot_predictions_vs_real(df_prediciones)
    g.savefig(rf"{path_dataout}/scatterplot_{pred_variable}_{model_name}.png", dpi=300)


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
            weights="imagenet",
        )
