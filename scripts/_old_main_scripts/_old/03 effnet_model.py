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

path_proyecto   =  "R:/Tesis Nico/Códigos"    # Ubicación de la carpeta del Proyecto
subproyecto = 'Testing - Deep Learning con MapTilesDownloader'

path_datain     = f"{path_proyecto}/data/data_in"
path_dataout    = f"{path_proyecto}/data/data_out/{subproyecto}"  # Bases procesadas por tus scripts
path_scripts    = f"{path_proyecto}/scripts/{subproyecto}"
path_figures    = f"{path_proyecto}/outputs/figures/{subproyecto}"
path_maps       = f"{path_proyecto}/outputs/maps/{subproyecto}"
path_tables     = f"{path_proyecto}/outputs/tables/{subproyecto}" 
from typing import Iterator, List, Union, Tuple, Any
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, Model, activations
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy, CategoricalAccuracy, CategoricalCrossentropy, TruePositives, FalsePositives, TrueNegatives, FalseNegatives
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNetV3Small, EfficientNetB0, EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2, EfficientNetV2B3, EfficientNetV2S
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


def create_generators(
    dir
    ) -> Tuple[Iterator, Iterator, Iterator]:
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
    
    train_generator = ImageDataGenerator(        
        rotation_range=5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=(0.75, 1),
        shear_range=0.1,
        zoom_range=[0.75, 1],
        horizontal_flip=True,
        vertical_flip=True,
    ) # create an ImageDataGenerator with multiple image augmentations
    
    validation_generator = ImageDataGenerator(
    )  # except for rescaling, no augmentations are needed for validation and testing generators
    test_generator = ImageDataGenerator()  # visualize image augmentations

    train_generator = train_generator.flow_from_directory(
        directory= f"{dir}\\train",
        batch_size=16,  # increase or decrease to fit your GPU  
        target_size=(224,224),
        class_mode='categorical'
    )

    validation_generator = validation_generator.flow_from_directory(
        directory=f"{dir}\\val",
        batch_size=128,  # increase or decrease to fit your GPU
        target_size=(224,224),
        class_mode='categorical'
    )

    test_generator = validation_generator

    return train_generator, validation_generator, test_generator


def get_callbacks(model_name: str) -> List[Union[TensorBoard, EarlyStopping, ModelCheckpoint]]:
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

    tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq = 1, profile_batch = '500,520')
    # use tensorboard --logdir logs/scalars in your command line to startup tensorboard with the correct logs

    early_stopping_callback = EarlyStopping(
        monitor="categorical_crossentropy",
        min_delta=1,  # model should improve by at least 1%
        patience=20,  # amount of epochs  with improvements worse than 0.5% until the model stops
        verbose=2,
        mode="min",
        restore_best_weights=True,  # restore the best model with the lowest validation error
    )

    model_checkpoint_callback = ModelCheckpoint(
        f"{path_dataout}/models/" + model_name + ".h5",
        monitor="categorical_crossentropy",
        verbose=0,
        save_best_only=True,  # save the best model
        mode="min",
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

    return [tensorboard_callback, early_stopping_callback, model_checkpoint_callback] #, confusion_matrix_callback]


def mobnet_v3_s() -> Sequential:  
    
    model = models.Sequential()
    model.add(MobileNetV3Small(weights=None, include_top=True, classes=10, input_shape=(224, 224, 3), minimalistic=True,
                               pooling='max', classifier_activation='softmax', include_preprocessing=True))
    return model

def effnet_b0() -> Sequential:  
    '''https://keras.io/api/applications/efficientnet_v2/#efficientnetv2s-function'''
    
    model = models.Sequential()
    # model.add(layers.Dropout(0.99, input_shape=(224, 224, 3)))
    model.add(EfficientNetB0(weights=None, include_top=True, classes=10, input_shape=(224, 224, 3),
                               pooling='max', classifier_activation='softmax'))
    return model

def effnet_v2_b0_reg() -> Sequential:  
    '''https://keras.io/api/applications/efficientnet_v2/#efficientnetv2s-function'''
    
    model = models.Sequential()
    model.add(EfficientNetV2B0(weights=None, include_top=False, input_shape=(224, 224, 3),
                               pooling='max', classifier_activation='softmax', include_preprocessing=True))
    model.add(layers.GlobalAveragePooling2D(name="avg_pool"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4, name="top_dropout"))
    model.add(layers.Dense(1, activation='linear', name='predictions'))
    
    return model

def run_model(
    model_name: str,
    model_function: Model,
    lr: float,
    train_generator: Iterator,
    validation_generator: Iterator,
    test_generator: Iterator,
    loss: str = 'categorical_cross_entropy',
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

    callbacks = get_callbacks(model_name)
    model = model_function
    model.summary()
    plot_model(model, to_file=model_name + ".png", show_shapes=True)

    radam = tfa.optimizers.RectifiedAdam(learning_rate=lr)
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
    optimizer = ranger

    model.compile(
        optimizer=optimizer, loss=loss, metrics=[CategoricalAccuracy(), CategoricalCrossentropy()]
    )
    history = model.fit(
        train_generator,
        epochs=300,
        validation_data=validation_generator,
        callbacks=callbacks,
        workers=8,  # adjust this according to the number of CPU cores of your machine
    )

    model.evaluate(
        test_generator, callbacks=callbacks,
    )
    return history  # type: ignore


# def plot_results(model_history_small_cnn: History, mean_baseline: float):
#     """This function uses seaborn with matplotlib to plot the trainig and validation losses of both input models in an
#     sns.relplot(). The mean baseline is plotted as a horizontal red dotted line.

#     Parameters
#     ----------
#     model_history_small_cnn : History
#         keras History object of the model.fit() method.
#     model_history_eff_net : History
#         keras History object of the model.fit() method.
#     mean_baseline : float
#         Result of the get_mean_baseline() function.
#     """

#     # create a dictionary for each model history and loss type
#     dict1 = {
#         "MSE": model_history_small_cnn.history["mean_squared_error"],
#         "type": "training",
#         "model": "small_cnn",
#     }
#     dict2 = {
#         "MSE": model_history_small_cnn.history["val_mean_squared_error"],
#         "type": "validation",
#         "model": "small_cnn",
#     }

#     # convert the dicts to pd.Series and concat them to a pd.DataFrame in the long format
#     s1 = pd.DataFrame(dict1)
#     s2 = pd.DataFrame(dict2)

#     df = pd.concat([s1, s2], axis=0).reset_index()
#     grid = sns.relplot(data=df, x=df["index"], y="MSE", hue="model", col="type", kind="line", legend=False)
#     # grid.set(ylim=(20, 100))  # set the y-axis limit
#     for ax in grid.axes.flat:
#         ax.axhline(
#             y=mean_baseline, color="lightcoral", linestyle="dashed"
#         )  # add a mean baseline horizontal bar to each plot
#         ax.set(xlabel="Epoch")
#     labels = ["small_cnn", "mean_baseline"]  # custom labels for the plot

#     plt.legend(labels=labels)
#     plt.savefig("training_validation.png")
#     plt.show()

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


def run(pred_variable: str, small_sample=False):
    """Run all the code of this file.

    Parameters
    ----------
    small_sample : bool, optional
        If you just want to check if the code is working, set small_sample to True, by default False
    """

    train_generator, validation_generator, test_generator = create_generators(
        dir=rf"R:\Tesis Nico\Códigos\data\data_in\rmax_deciles_224"
    )

    small_cnn_history = run_model(
        model_name=f"effnet_b0_{pred_variable}",
        model_function=effnet_b0(),
        lr=0.005,         # lr=0.00005 para v0
        train_generator=train_generator,
        validation_generator=validation_generator,
        test_generator=test_generator,
        loss = "categorical_crossentropy",
    )

    # plot_results(small_cnn_history, mean_baseline)
    # predictions = predict_values(densenet_model(), test_generator)
    # print(predictions)
    

if __name__ == "__main__":
    # run(pred_variable='rmax_d', small_sample=False)
    # run_densenet(pred_variable='rmax_d', small_sample=5000)
    
    for col in ['rmax_d']:
        run(pred_variable=col, small_sample=False)