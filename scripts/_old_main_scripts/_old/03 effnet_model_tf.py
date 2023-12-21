### TODO:
# Intentar que el he_normal funcione. Posible problema de los bias en cero: https://www.kaggle.com/code/sauravjoshi23/weight-initialization-methods-keras
#   kernel_initializer='he_normal', bias_initializer='zeros'

# Correr con modelo small_cnn bueno:
# Solo AMBA:
# - pm2
# - Ingreso esperado
# - icv2010
# - Correlacionar RMax_pred con ICPAG
# Poner a descargar alfombra de mapa de 100x100

##############      Configuración      ##############
from dotenv import dotenv_values
from plantilla import plantilla

path_proyecto = "R:/Tesis Nico/Códigos"  # Ubicación de la carpeta del Proyecto
subproyecto = "Testing - Deep Learning con MapTilesDownloader"

path_datain = f"{path_proyecto}/data/data_in"
path_dataout = f"{path_proyecto}/data/data_out/{subproyecto}"  # Bases procesadas por tus scripts
path_scripts = f"{path_proyecto}/scripts/{subproyecto}"
path_figures = f"{path_proyecto}/outputs/figures/{subproyecto}"
path_maps = f"{path_proyecto}/outputs/maps/{subproyecto}"
path_tables = f"{path_proyecto}/outputs/tables/{subproyecto}"
import prediction_tools

from typing import Iterator, List, Union, Tuple, Any
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split

from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, Model, activations
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras import losses
from tensorflow.keras.metrics import (
    CategoricalAccuracy,
    CategoricalCrossentropy,
    MeanAbsoluteError,
    MeanSquaredError,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import (
    MobileNetV3Small,
    EfficientNetB0,
    EfficientNetV2B0,
)
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import History
from densenet import densenet_model

# the next 3 lines of code are for my machine and setup due to https://github.com/tensorflow/tensorflow/issues/43174
import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# def get_mean_baseline(pred_variable: str, train: pd.DataFrame, val: pd.DataFrame) -> float:
#     """Calculates the mean MAE and MAPE baselines by taking the mean values of the training data as prediction for the
#     validation target feature.

#     Parameters
#     ----------
#     train : pd.DataFrame
#         Pandas DataFrame containing your training data.
#     val : pd.DataFrame
#         Pandas DataFrame containing your validation data.

#     Returns
#     -------
#     float
#         MAPE value.
#     """
#     y_hat = train[pred_variable].mean()
#     val["y_hat"] = y_hat
#     mae = MeanSquaredError()
#     mae = mae(val[pred_variable], val["y_hat"]).numpy()  # type: ignore
#     mape = MeanSquaredLogarithmicError()
#     mape = mape(val[pred_variable], val["y_hat"]).numpy()  # type: ignore

#     print(mae)
#     print("mean baseline MSE: ", mape)

#     return mape


def create_datasets(variable, kind, small_sample=False) -> Tuple[Iterator, Iterator, Iterator]:
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
    Tuple[Iterator, Iterator, Iterator]
        keras ImageDataGenerators used for training, validating and testing of your models.
    """
    if small_sample:
        print("Small sample not implemented yet")

    # Create a data augmentation stage with horizontal flipping, rotations, zooms
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical", seed=825, input_shape=(224, 224, 3)),
            layers.RandomRotation(0.1, fill_mode="reflect", seed=825),
            layers.RandomZoom(0.1, 0.1, "reflect", seed=825),
            layers.RandomContrast(0.1, seed=825),
            layers.RandomBrightness(0.1, seed=825),
        ],
        name="data_augmentation",
    )

    # Open dataframe with files and labels
    df = pd.read_parquet(rf"{path_dataout}\dataset_amba.parquet")
    df = df.dropna(subset=[variable] + ["image"]).reset_index(drop=True)
    if small_sample:
        df = df.sample(100, random_state=825).reset_index(drop=True)

    # Get list of filenames and corresponding list of labels
    print(len(df["image"].to_list()))
    print(len(df[variable].to_list()))

    filenames = tf.constant(df["image"].to_list())
    if kind == "reg":
        labels = tf.constant(df[variable].to_list())
    elif kind == "cla":
        labels = tf.one_hot(
            (df[variable] - 1).to_list(), 10
        )  # Fist class corresponds to decile 1, etc.

    # Train, validation and test split
    train_size = round(0.8 * len(filenames))
    val_size = round(0.1 * len(filenames))
    test_size = round(0.1 * len(filenames))

    # Create tf.data pipeline
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    # Parse every image in the dataset using `map`
    def img_file_to_tensor(file, label):
        def _img_file_to_tensor(file, label):
            im = tf.image.decode_jpeg(tf.io.read_file(file), channels=3)
            im = tf.image.resize(im, [224, 224])
            im = tf.cast(im, tf.float32)
            label = tf.cast(label, tf.float32)
            return im, label

        return tf.py_function(_img_file_to_tensor, [file, label], [tf.float32, tf.float32])

    dataset = dataset.map(img_file_to_tensor)

    # Split dataset
    def get_dataset_partitions_tf(
        ds,
        ds_size,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        shuffle=True,
        shuffle_size=10000,
    ):
        assert (train_split + test_split + val_split) == 1

        if shuffle:
            # Specify seed to always have the same split distribution between runs
            ds = ds.shuffle(shuffle_size, seed=825)

        train_size = int(train_split * ds_size)
        val_size = int(val_split * ds_size)

        train_ds = ds.take(train_size)
        val_ds = ds.skip(train_size).take(val_size)
        test_ds = ds.skip(train_size).skip(val_size).take(test_size)

        return train_ds, val_ds, test_ds

    train_dataset, val_dataset, test_dataset = get_dataset_partitions_tf(
        dataset,
        len(filenames),
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        shuffle=True,
        shuffle_size=round(len(filenames) / 5),
    )

    # Prepare dataset for training
    train_dataset = (
        train_dataset.batch(16)
        .map(lambda x, y: (data_augmentation(x), y))
        .prefetch(tf.data.AUTOTUNE)
    )
    val_dataset = val_dataset.batch(128)

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
        "logs/scalars/" + model_name + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )  # create a folder for each model.

    tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=1, profile_batch="500,520")
    # use tensorboard --logdir logs/scalars in your command line to startup tensorboard with the correct logs

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.5,  # model should improve by at least 1%
        patience=3,  # amount of epochs  with improvements worse than 0.5% until the model stops
        verbose=2,
        mode="auto",
        restore_best_weights=True,  # restore the best model with the lowest validation error
    )

    model_checkpoint_callback = ModelCheckpoint(
        f"{path_dataout}/models/" + model_name + ".h5",
        monitor='val_loss',
        verbose=0,
        save_best_only=True,  # save the best model
        mode="auto",
        save_freq="epoch",  # save every epoch
    )  # saving eff_net takes quite a bit of time

    # def log_confusion_matrix(epoch, logs):
    #     import numpy as np
    #     from sklearn import metrics

    #     def plot_confusion_matrix(cm, class_names):
    #         import itertools

    #         figure = plt.figure(figsize=(8, 8))
    #         plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Accent)
    #         plt.title("Confusion matrix")
    #         plt.colorbar()
    #         tick_marks = np.arange(len(class_names))
    #         plt.xticks(tick_marks, class_names, rotation=45)
    #         plt.yticks(tick_marks, class_names)

    #         cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    #         threshold = cm.max() / 2.

    #         for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #             color = "white" if cm[i, j] > threshold else "black"
    #             plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    #         plt.tight_layout()
    #         plt.ylabel('True label')
    #         plt.xlabel('Predicted label')

    #         return figure

    #     def plot_to_image(figure):

    #         import io

    #         buf = io.BytesIO()
    #         plt.savefig(buf, format='png')
    #         plt.close(figure)
    #         buf.seek(0)

    #         digit = tf.image.decode_png(buf.getvalue(), channels=4)
    #         digit = tf.expand_dims(digit, 0)

    #         return digit

    #     predictions = model.predict(X_test)
    #     predictions = np.argmax(predictions, axis=1)

    #     cm = metrics.confusion_matrix(y_test, predictions)
    #     figure = plot_confusion_matrix(cm, class_names=class_names)
    #     cm_image = plot_to_image(figure)

    #     with file_writer_cm.as_default():
    #         tf.summary.image("Confusion Matrix", cm_image, step=epoch)

    # confusion_matrix_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

    return [
        tensorboard_callback,
        early_stopping_callback,
        model_checkpoint_callback,
    ]  # , confusion_matrix_callback]


def mobnet_v3_s() -> Sequential:

    model = models.Sequential()
    model.add(
        MobileNetV3Small(
            weights=None,
            include_top=True,
            classes=10,
            input_shape=(224, 224, 3),
            minimalistic=True,
            pooling="max",
            classifier_activation="softmax",
            include_preprocessing=True,
        )
    )
    return model


def effnet_b0(kind="cla") -> Sequential:
    """https://keras.io/api/applications/efficientnet_v2/#efficientnetv2s-function"""

    assert kind in ["cla", "reg"], "kind must be either cla or reg"

    model = models.Sequential()
    if kind == "cla":
        model.add(
            EfficientNetB0(
                weights=None,
                include_top=True,
                classes=10,
                input_shape=(224, 224, 3),
                pooling="max",
                classifier_activation="softmax",
            )
        )

    if kind == "reg":
        model.add(
            EfficientNetB0(
                weights=None,
                include_top=False,
                input_shape=(224, 224, 3),
            )
        )
        model.add(layers.GlobalAveragePooling2D(name="avg_pool"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4, name="top_dropout"))
        model.add(layers.Dense(1, activation="linear", name="predictions"))

    return model


def effnet_v2_b0_reg() -> Sequential:
    """https://keras.io/api/applications/efficientnet_v2/#efficientnetv2s-function"""

    model = models.Sequential()
    model.add(
        EfficientNetV2B0(
            weights=None,
            include_top=False,
            input_shape=(224, 224, 3),
            pooling="max",
            classifier_activation="softmax",
            include_preprocessing=True,
        )
    )
    model.add(layers.GlobalAveragePooling2D(name="avg_pool"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4, name="top_dropout"))
    model.add(layers.Dense(1, activation="linear", name="predictions"))

    return model


def run_model(
    model_name: str,
    model_function: Model,
    lr: float,
    train_generator: Iterator,
    validation_generator: Iterator,
    test_generator: Iterator,
    loss: str,
    metrics: List[str],
) -> History:
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
        The history of the keras model as a History object. To access it as a Dict, use history.history. For an example
        see plot_results().
    """

    callbacks = get_callbacks(model_name, loss)
    model = model_function
    model.summary()
    plot_model(model, to_file=model_name + ".png", show_shapes=True)

    radam = tfa.optimizers.RectifiedAdam(learning_rate=lr)
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
    optimizer = ranger

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    history = model.fit(
        train_generator,
        epochs=1,
        validation_data=validation_generator,
        shuffle=True,
        callbacks=callbacks,
        workers=8,  # adjust this according to the number of CPU cores of your machine
    )

    model.evaluate(
        test_generator,
        callbacks=callbacks,
    )
    return model, history  # type: ignore


def predict_values(model: Model, test_generator: Iterator) -> pd.DataFrame:
    """This function predicts the values for the test set and returns a pandas DataFrame with the predictions.

    Parameters
    ----------
    model_name : str
        The name of the model as a string.
    model : Model
        Keras model function like small_cnn()  or adapt_efficient_net().
    test_generator : Iterator
        keras ImageDataGenerators for the test data.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with the predictions.
    """

    predictions = model.predict(test_generator)
    predictions = pd.DataFrame(predictions, columns=["prediction"])
    return predictions


def run(pred_variable: str, kind: str, small_sample=False):
    """Run all the code of this file.

    Parameters
    ----------
    small_sample : bool, optional
        If you just want to check if the code is working, set small_sample to True, by default False
    """
    assert kind in ["reg", "cla"], "kind must be either 'reg' or 'cla'"

    model_name = f"effnet_b0_{pred_variable}"
    if kind == "reg":
        loss = keras.losses.MeanSquaredError()
        metrics = [keras.metrics.MeanAbsoluteError(), keras.metrics.MeanSquaredError()]

    elif kind == "cla":
        loss = keras.losses.CategoricalCrossentropy()
        metrics = [keras.metrics.CategoricalAccuracy(), keras.metrics.CategoricalCrossentropy()]

    train_dataset, test_dataset, val_dataset, filenames = create_datasets(
        variable=pred_variable,
        kind=kind,
        small_sample=small_sample,
    )
    print("train_dataset:", len(list(train_dataset)))
    print("test_dataset:", len(list(test_dataset)))
    print("val_dataset:", len(list(val_dataset)))

    model, history = run_model(
        model_name=model_name,
        model_function=effnet_b0(kind),
        lr=0.005,  # lr=0.00005 para v0
        train_generator=train_dataset,
        validation_generator=val_dataset,
        test_generator=test_dataset,
        loss=loss,
        metrics=metrics,
    )

    # plot_results(small_cnn_history, mean_baseline)
    predicted_array = prediction_tools.get_predicted_values(model, test_dataset, kind="reg")
    real_array = prediction_tools.get_real_values(test_dataset)

    # for norm in ["pred", "true"]:
    #     prediction_tools.plot_confusion_matrix(
    #         real_array,
    #         predicted_array,
    #         model_name,
    #         normalize=norm,
    #     )
    #     fig.savefig(f"{path_figures}/confusion_matrix_{model_name}_{norm}.png")

    # Creo dataframe para exportar:
    d = {"real": df["RMax (Real Value)"].to_numpy(), "pred": predicted_values}
    pd.DataFrame(data=d).to_parquet(f"{path_dataout}/preds_{model_name}.parquet")


if __name__ == "__main__":

    variable_models = {
        # "rmax_d": "cla",
        # "rmin_d": "cla",
        # "icv2010_d": "cla",
        "rmax": "reg",
        # "rmin": "reg",
        # "icv2010": "reg",
        # "nbi_rc_val": "reg",
        # "pm2": "reg",
        # "viv_part": "reg",
    }

    for var, kind in variable_models.items():
        print("#######################################################")
        print(f"Running model for {var} ({kind})")
        print("#######################################################")
        run(pred_variable=var, kind=kind, small_sample=True)
