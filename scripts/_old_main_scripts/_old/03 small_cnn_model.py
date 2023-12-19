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
from tensorflow.keras.losses import MeanSquaredError, MeanSquaredLogarithmicError, MeanAbsoluteError, CategoricalCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import History
from densenet import densenet_model

# the next 3 lines of code are for my machine and setup due to https://github.com/tensorflow/tensorflow/issues/43174
import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def visualize_augmentations(data_generator: ImageDataGenerator, df: pd.DataFrame):
    """Visualizes the keras augmentations with matplotlib in 3x3 grid. This function is part of create_generators() and
    can be accessed from there.

    Parameters
    ----------
    data_generator : Iterator
        The keras data generator of your training data.
    df : pd.DataFrame
        The Pandas DataFrame containing your training data.
    """
    # super hacky way of creating a small dataframe with one image
    series = df.iloc[2]

    df_augmentation_visualization = pd.concat([series, series], axis=1).transpose()

    iterator_visualizations = data_generator.flow_from_dataframe(  # type: ignore
        dataframe=df_augmentation_visualization,
        x_col="image",
        y_col=pred_variable,
        class_mode="raw",
        target_size=(244, 244),  # size of the image
        batch_size=4,  # use only one image for visualization
    )

    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)  # create a 3x3 grid
        batch = next(iterator_visualizations)  # get the next image of the generator (always the same image)
        img = batch[0]  # type: ignore
        img = img[0, :, :, :]  # remove one dimension for plotting without issues
        plt.imshow(img)
    plt.show()
    plt.close()


def get_mean_baseline(pred_variable: str, train: pd.DataFrame, val: pd.DataFrame) -> float:
    """Calculates the mean MAE and MAPE baselines by taking the mean values of the training data as prediction for the
    validation target feature.

    Parameters
    ----------
    train : pd.DataFrame
        Pandas DataFrame containing your training data.
    val : pd.DataFrame
        Pandas DataFrame containing your validation data.

    Returns
    -------
    float
        MAPE value.
    """
    y_hat = train[pred_variable].mean()
    val["y_hat"] = y_hat
    mae = MeanSquaredError()
    mae = mae(val[pred_variable], val["y_hat"]).numpy()  # type: ignore
    mape = MeanSquaredLogarithmicError()
    mape = mape(val[pred_variable], val["y_hat"]).numpy()  # type: ignore

    print(mae)
    print("mean baseline MSE: ", mape)

    return mape


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Accepts a Pandas DataFrame and splits it into training, testing and validation data. Returns DataFrames.

    Parameters
    ----------
    df : pd.DataFrame
        Your Pandas DataFrame containing all your data.

    Returns
    -------
    Union[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        [description]
    """
    train, val = train_test_split(df, test_size=0.2, random_state=1)  # split the data with a validation size o 20%
    train, test = train_test_split(
        train, test_size=0.125, random_state=1
    )  # split the data with an overall  test size of 10%

    print("shape train: ", train.shape)  # type: ignore
    print("shape val: ", val.shape)  # type: ignore
    print("shape test: ", test.shape)  # type: ignore

    print("Descriptive statistics of train:")
    print(train.describe())  # type: ignore
    return train, val, test  # type: ignore


def create_generators(
    pred_variable: str, df: pd.DataFrame, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, plot_augmentations: Any
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
    
    def crop(image):
        ''' Custom function to crop the image to a square shape of 244x244. '''      
        # start_x = (512 - 267/2)
        # start_y = (512 - 267/2)
        # cropped_image=image[start_x:(512 - start_x), start_y:(512 - start_y), :]
        cropped_image=image[134:378, 134:378, :]
        return cropped_image
    
    train_generator = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=(0.75, 1),
        shear_range=0.1,
        zoom_range=[0.75, 1],
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2,
    )  # create an ImageDataGenerator with multiple image augmentations
    validation_generator = ImageDataGenerator(
        rescale=1.0 / 255,
    )  # except for rescaling, no augmentations are needed for validation and testing generators
    test_generator = ImageDataGenerator(
        rescale=1.0 / 255,
    )
    # visualize image augmentations
    if plot_augmentations == True:
        pass #visualize_augmentations(train_generator, df)

    train_generator = train_generator.flow_from_dataframe(
        dataframe=train,
        x_col="image",  # this is where your image data is stored
        y_col=pred_variable,  # this is your target feature
        class_mode="raw",  # use "raw" for regressions
        target_size=(512, 512),
        batch_size=4,  # increase or decrease to fit your GPU
    )

    validation_generator = validation_generator.flow_from_dataframe(
        dataframe=val, x_col="image", y_col=pred_variable, class_mode="raw", target_size=(512, 512), batch_size=128,
    )
    test_generator = test_generator.flow_from_dataframe(
        dataframe=test, x_col="image", y_col=pred_variable, class_mode="raw", target_size=(512, 512), batch_size=128,
    )

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
        monitor="val_mean_squared_logarithmic_error",
        min_delta=1,  # model should improve by at least 1%
        patience=20,  # amount of epochs  with improvements worse than 0.5% until the model stops
        verbose=2,
        mode="min",
        restore_best_weights=True,  # restore the best model with the lowest validation error
    )

    model_checkpoint_callback = ModelCheckpoint(
        f"{path_dataout}/models/" + model_name + ".h5",
        monitor="val_mean_squared_logarithmic_error",
        verbose=0,
        save_best_only=True,  # save the best model
        mode="min",
        save_freq="epoch",  # save every epoch
    )  # saving eff_net takes quite a bit of time
    return [tensorboard_callback, early_stopping_callback, model_checkpoint_callback]


# def small_cnn() -> Sequential:
    """A very small custom convolutional neural network with image input dimensions of 224x224x3.

    Returns
    -------
    Sequential
        The keras Sequential model.
    """
    # V1: Total params: 13,364,353
    # model = models.Sequential()
    # model.add(layers.CenterCrop(244, 244, input_shape=(512, 512, 3)))
    # model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), activation="relu"))

    # model.add(layers.Flatten())
    # model.add(layers.Dense(64, activation="relu"))
    # model.add(layers.Dense(1, activation="sigmoid"))

    # # V2: Total params: 28,957,825
    # model = models.Sequential()
    # model.add(layers.CenterCrop(244, 244, input_shape=(512, 512, 3)))
    # model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), activation="relu"))

    # model.add(layers.Flatten())
    # model.add(layers.Dense(64, activation="relu"))
    # model.add(layers.Dense(1, activation="sigmoid"))

    # V4: Total params: 12,938,433 (14,838,977 genera OOM - sin el crop)
    model = models.Sequential()
    model.add(layers.CenterCrop(300, 300, input_shape=(512, 512, 3)))
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))

    # V3: Total params: 63,036,545
    # model = models.Sequential()
    # model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(512, 512, 3)))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), activation="relu"))

    # model.add(layers.Flatten())
    # model.add(layers.Dense(64, activation="relu"))
    # model.add(layers.Dense(1, activation="sigmoid"))


    return model

def small_cnn_v0() -> Sequential:
    """ A very small custom convolutional neural network with image input dimensions of 224x224x3, cropped from 512x512x3.
    
    _________________________________________________________________
    Layer (type)                Output Shape              Param #
    =================================================================
    center_crop (CenterCrop)    (None, 244, 244, 3)       0

    conv2d (Conv2D)             (None, 242, 242, 32)      896

    max_pooling2d (MaxPooling2D  (None, 121, 121, 32)     0
    )

    conv2d_1 (Conv2D)           (None, 119, 119, 64)      18496

    max_pooling2d_1 (MaxPooling  (None, 59, 59, 64)       0
    2D)

    conv2d_2 (Conv2D)           (None, 57, 57, 64)        36928

    flatten (Flatten)           (None, 207936)            0

    dense (Dense)               (None, 64)                13307968

    dense_1 (Dense)             (None, 1)                 65

    =================================================================
    Total params: 13,364,353
    Trainable params: 13,364,353
    Non-trainable params: 0
    _________________________________________________________________
    
    Returns
    -------
    Sequential
        The keras Sequential model.
    """
    
    model = models.Sequential()
    model.add(layers.CenterCrop(244, 244, input_shape=(512, 512, 3))) # Maybe era (244, 244)
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu")) 

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

def small_cnn_v0_1() -> Sequential:
    """ A very small custom convolutional neural network with image input dimensions of 224x224x3.
    
    Hago que la conv final tenga 128 a ver si entra y mejora. Agrego initializers para que no explote
    Returns
    -------
    Sequential
        The keras Sequential model.
    """
    
    initializer = tf.keras.initializers.he_uniform()

    model = models.Sequential()
    model.add(layers.CenterCrop(300, 300, input_shape=(512, 512, 3)))
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu")) 

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

def small_cnn_v0_reduc() -> Sequential:
    """ A very small custom convolutional neural network with image input dimensions of 224x224x3.
    
    Hago que la conv final tenga 128 a ver si entra y mejora. Agrego initializers para que no explote
    Returns
    -------
    Sequential
        The keras Sequential model.
    """
    
    initializer = tf.keras.initializers.he_uniform()

    model = models.Sequential()
    model.add(layers.Resizing(128, 128, input_shape=(512, 512, 3)))
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu")) 

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(1, activation='linear'))

    return model

def small_cnn_v1() -> Sequential:
    """ A very small custom convolutional neural network with image input dimensions of 224x224x3.
        Con el relu con max en 1.
        UNA VERGA. No aprendió nada.
    Returns
    -------
    Sequential
        The keras Sequential model.
    """
    from keras import backend as K
    def create_relu_advanced(max_value=1.):        
        def relu_advanced(x):
            return K.relu(x, max_value=K.cast_to_floatx(max_value))
        return relu_advanced

    model = models.Sequential()
    model.add(layers.CenterCrop(300, 300, input_shape=(512, 512, 3)))
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu")) 

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(1, activation=create_relu_advanced(max_value=1.)))

    return model

def small_cnn_v1_1() -> Sequential:
    """ A very small custom convolutional neural network with image input dimensions of 224x224x3.
        Con el relu con max en 1 + LeakyRelu.
        UNA VERGA. No aprendió nada.
    Returns
    -------
    Sequential
        The keras Sequential model.
    """
    from keras import backend as K
    def create_relu_advanced(max_value=1.):        
        def relu_advanced(x):
            return K.relu(x, max_value=K.cast_to_floatx(max_value))
        return relu_advanced

    from keras.layers import LeakyReLU

    model = models.Sequential()
    model.add(layers.CenterCrop(300, 300, input_shape=(512, 512, 3)))
    model.add(layers.Conv2D(32, (3, 3), activation=LeakyReLU(alpha=0.05)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation=LeakyReLU(alpha=0.05)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation=LeakyReLU(alpha=0.05))) 

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation=LeakyReLU(alpha=0.05)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(1, activation=create_relu_advanced(max_value=1.)))

    return model

def AlexNet() -> Sequential:
    """ AlexNet (Layer normalization + Capa Conv añadida entre cada capa y su activación. 
    https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py"""
    
    model = models.Sequential()
    model.add(layers.RandomCrop(244, 244, input_shape=(512, 512, 3)))
    
    model.add(layers.Conv2D(96, kernel_size=(11,11), strides= 4,
                            padding= 'valid',
                            kernel_initializer= 'he_normal'))
    model.add(layers.LayerNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(3,3), strides= (2,2),
                            padding= 'valid', data_format= None))

    model.add(layers.Conv2D(256, kernel_size=(5,5), strides= 1,
                    padding= 'same',
                    kernel_initializer= 'he_normal'))
    model.add(layers.LayerNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(3,3), strides= (2,2),
                            padding= 'valid', data_format= None)) 

    model.add(layers.Conv2D(384, kernel_size=(3,3), strides= 1,
                    padding= 'same',
                    kernel_initializer= 'he_normal'))
    model.add(layers.LayerNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(384, kernel_size=(3,3), strides= 1,
                    padding= 'same',
                    kernel_initializer= 'he_normal'))
    model.add(layers.LayerNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(256, kernel_size=(3,3), strides= 1,
                    padding= 'same',
                    kernel_initializer= 'he_normal'))
    model.add(layers.LayerNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(3,3), strides= (2,2),
                            padding= 'valid', data_format= None))


    model.add(tfa.layers.AdaptiveAveragePooling2D((6,6)))
    model.add(layers.Dropout(rate=0.5))

    model.add(layers.Dense(4096, activation='linear', kernel_initializer= 'he_normal'))
    model.add(layers.LayerNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(rate=0.5))
    
    model.add(layers.Dense(4096, activation='linear', kernel_initializer= 'he_normal'))
    model.add(layers.Activation('relu'))
    model.add(layers.LayerNormalization())

    model.add(layers.Dense(1, activation='linear', kernel_initializer= 'he_normal'))

    return model

def small_cnn_v3() -> Sequential:
    """ Batch normalization entre cada capa y su activación. """
    

    model = models.Sequential()
    model.add(layers.RandomCrop(244, 244, input_shape=(512, 512, 3)))
    
    model.add(layers.Conv2D(32, (3, 3), activation='linear'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='linear'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='linear')) 
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='linear'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

def small_cnn_v3_1_reduc() -> Sequential:
    """ layer normalization entre cada capa y su activación. Batch norm no funca
        porque uso batches de 1, se supone que no funciona bien para muestras de 
        menos de 32 (demasiada varianza en las estadísticas de cada batch).
        
        'there are strong theoretical reasons against it, and multiple publications 
        have shown BN performance degrade for batch_size under 32, and severely for <=8. 
        In a nutshell, batch statistics "averaged" over a single sample vary greatly 
        sample-to-sample (high variance), and BN mechanisms don't work as intended'
        (https://stackoverflow.com/questions/59648509/batch-normalization-when-batch-size-1).
        
        Layer normalization is independent of the batch size, so it can be applied to 
        batches with smaller sizes as well.
        (https://www.pinecone.io/learn/batch-layer-normalization/)"""
    

    model = models.Sequential()
    model.add(layers.Resizing(128, 128, input_shape=(512, 512, 3)))
    
    model.add(layers.Conv2D(32, (3, 3), activation='linear'))
    model.add(layers.LayerNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='linear'))
    model.add(layers.LayerNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='linear')) 
    model.add(layers.LayerNormalization())
    model.add(layers.Activation('relu'))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='linear'))
    model.add(layers.LayerNormalization())
    model.add(layers.Activation('relu'))
    
    model.add(layers.Dense(1, activation='linear'))

    return model


def small_cnn_v3_1_reduc() -> Sequential:
    """ layer normalization entre cada capa y su activación. Batch norm no funca
        porque uso batches de 1, se supone que no funciona bien para muestras de 
        menos de 32 (demasiada varianza en las estadísticas de cada batch).
        
        'there are strong theoretical reasons against it, and multiple publications 
        have shown BN performance degrade for batch_size under 32, and severely for <=8. 
        In a nutshell, batch statistics "averaged" over a single sample vary greatly 
        sample-to-sample (high variance), and BN mechanisms don't work as intended'
        (https://stackoverflow.com/questions/59648509/batch-normalization-when-batch-size-1).
        
        Layer normalization is independent of the batch size, so it can be applied to 
        batches with smaller sizes as well.
        (https://www.pinecone.io/learn/batch-layer-normalization/)"""
    

    model = models.Sequential()
    model.add(layers.Resizing(128, 128, input_shape=(512, 512, 3)))
    
    model.add(layers.Conv2D(32, (3, 3), activation='linear'))
    model.add(layers.LayerNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='linear'))
    model.add(layers.LayerNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='linear')) 
    model.add(layers.LayerNormalization())
    model.add(layers.Activation('relu'))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='linear'))
    model.add(layers.LayerNormalization())
    model.add(layers.Activation('relu'))
    
    model.add(layers.Dense(1, activation='linear'))

    return model

def run_model(
    model_name: str,
    model_function: Model,
    lr: float,
    train_generator: Iterator,
    validation_generator: Iterator,
    test_generator: Iterator,
    loss: str = 'mean_squared_error',
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
        optimizer=optimizer, loss=loss, metrics=[MeanAbsoluteError(), MeanSquaredError(), MeanSquaredLogarithmicError()]
        # optimizer=optimizer, loss="mean_squared_error", metrics=[MeanAbsoluteError(), CategoricalCrossentropy(), MeanSquaredError(), MeanSquaredLogarithmicError()]
    )
    history = model.fit(
        train_generator,
        epochs=100,
        validation_data=validation_generator,
        callbacks=callbacks,
        workers=8,  # adjust this according to the number of CPU cores of your machine
    )

    model.evaluate(
        test_generator, callbacks=callbacks,
    )
    return history  # type: ignore


def plot_results(model_history_small_cnn: History, mean_baseline: float):
    """This function uses seaborn with matplotlib to plot the trainig and validation losses of both input models in an
    sns.relplot(). The mean baseline is plotted as a horizontal red dotted line.

    Parameters
    ----------
    model_history_small_cnn : History
        keras History object of the model.fit() method.
    model_history_eff_net : History
        keras History object of the model.fit() method.
    mean_baseline : float
        Result of the get_mean_baseline() function.
    """

    # create a dictionary for each model history and loss type
    dict1 = {
        "MSE": model_history_small_cnn.history["mean_squared_error"],
        "type": "training",
        "model": "small_cnn",
    }
    dict2 = {
        "MSE": model_history_small_cnn.history["val_mean_squared_error"],
        "type": "validation",
        "model": "small_cnn",
    }

    # convert the dicts to pd.Series and concat them to a pd.DataFrame in the long format
    s1 = pd.DataFrame(dict1)
    s2 = pd.DataFrame(dict2)

    df = pd.concat([s1, s2], axis=0).reset_index()
    grid = sns.relplot(data=df, x=df["index"], y="MSE", hue="model", col="type", kind="line", legend=False)
    # grid.set(ylim=(20, 100))  # set the y-axis limit
    for ax in grid.axes.flat:
        ax.axhline(
            y=mean_baseline, color="lightcoral", linestyle="dashed"
        )  # add a mean baseline horizontal bar to each plot
        ax.set(xlabel="Epoch")
    labels = ["small_cnn", "mean_baseline"]  # custom labels for the plot

    plt.legend(labels=labels)
    plt.savefig("training_validation.png")
    plt.show()

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

    df = pd.read_parquet(f"{path_dataout}/dataset_indicadores_rc_con_img.parquet")
    
    if small_sample:
        df = df.sample(small_sample)  # set small_sampe to True if you want to check if your code works without long waiting
    train, val, test = split_data(df)  # split your data
    mean_baseline = get_mean_baseline(pred_variable, train, val)
    train_generator, validation_generator, test_generator = create_generators(
        pred_variable=pred_variable, df=df, train=train, val=val, test=test, plot_augmentations=True
    )

    small_cnn_history = run_model(
        model_name=f"small_cnn_v3_1_reduc_{pred_variable}",
        model_function=small_cnn_v3_1_reduc(),
        lr=0.00005,         # lr=0.00005 para v0
        train_generator=train_generator,
        validation_generator=validation_generator,
        test_generator=test_generator,
        loss = "mean_squared_error",
    )

    # plot_results(small_cnn_history, mean_baseline)
    # predictions = predict_values(densenet_model(), test_generator)
    # print(predictions)
    
def run_densenet(pred_variable: str, small_sample=False):
    """Run all the code of this file.

    Parameters
    ----------
    small_sample : bool, optional
        If you just want to check if the code is working, set small_sample to True, by default False
    """

    df = pd.read_parquet(f"{path_dataout}/dataset_indicadores_rc_con_img.parquet")
    
    if small_sample:
        df = df.sample(small_sample)  # set small_sampe to True if you want to check if your code works without long waiting
    
    df = pd.get_dummies(df, columns=[pred_variable])

    train, val, test = split_data(df)  # split your data
    print(df[[col for col in df.columns if pred_variable in col]].shape)
    train_generator, validation_generator, test_generator = create_generators(
        pred_variable=[col for col in df.columns if pred_variable in col], df=df, train=train, val=val, test=test, plot_augmentations=True
    )

    small_cnn_history = run_model(
        model_name=f"densenet_{pred_variable}",
        model_function=densenet_model(classes=10, batch_size=1, shape=(512, 512, 3)), # Default: resnet121
        lr=0.0001,         # lr=0.00005 para v0
        train_generator=train_generator,
        validation_generator=validation_generator,
        test_generator=test_generator,
        loss = 'categorical_crossentropy',
    )

    # plot_results(small_cnn_history, mean_baseline)
    # predictions = predict_values(densenet_model(), test_generator)
    # print(predictions)

def run_effnet(pred_variable: str, small_sample=False):
    """Run all the code of this file with EfficientNetV2.

    Parameters
    ----------
    small_sample : bool, optional
        If you just want to check if the code is working, set small_sample to True, by default False
    """

    df = pd.read_parquet(f"{path_dataout}/dataset_indicadores_rc_con_img.parquet")
    
    if small_sample:
        df = df.sample(small_sample)  # set small_sampe to True if you want to check if your code works without long waiting
    
    df = pd.get_dummies(df, columns=[pred_variable])

    train, val, test = split_data(df)  # split your data
    print(df[[col for col in df.columns if pred_variable in col]].shape)
    train_generator, validation_generator, test_generator = create_generators(
        pred_variable=[col for col in df.columns if pred_variable in col], df=df, train=train, val=val, test=test, plot_augmentations=True
    )

    small_cnn_history = run_model(
        model_name=f"densenet_{pred_variable}",
        model_function=densenet_model(classes=10, batch_size=1, shape=(512, 512, 3)), # Default: resnet121
        lr=0.0001,         # lr=0.00005 para v0
        train_generator=train_generator,
        validation_generator=validation_generator,
        test_generator=test_generator,
        loss = 'categorical_crossentropy',
    )

    # plot_results(small_cnn_history, mean_baseline)
    # predictions = predict_values(densenet_model(), test_generator)
    # print(predictions)

if __name__ == "__main__":
    # run(pred_variable='rmax_d', small_sample=False)
    # run_densenet(pred_variable='rmax_d', small_sample=5000)
    
    for col in ['rmin', 'aa_mean_mean']:
        run(pred_variable=col, small_sample=False)