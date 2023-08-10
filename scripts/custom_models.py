import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

from tensorflow.keras import layers, models, Model
from tensorflow.python.keras.callbacks import (
    TensorBoard,
    EarlyStopping,
    ModelCheckpoint,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import (
    MobileNetV2,
    MobileNetV3Small,
    EfficientNetB0,
    EfficientNetV2B0,
    EfficientNetV2S,
    EfficientNetV2L,
    ResNet152V2,
)


def rebuild_top(model_base, kind="cla") -> Sequential:
    """Rebuild top of a pre-trained model to make it suitable for classification or regression."""

    assert kind in ["cla", "reg"], "kind must be either cla or reg"

    model = tf.keras.Sequential()

    model.add(model_base)

    # Rebuild top
    model.add(layers.GlobalAveragePooling2D(name="avg_pool"))
    model.add(layers.BatchNormalization())
    if kind == "cla":
        # Add fully conected layers
        # model.add(layers.Dense(2048, name="fc1", activation="relu"))
        #         model.add(layers.Dense(2048, name="fc1", activation="relu"))
        model.add(layers.Dense(10, name="predictions", activation="softmax"))
    if kind == "reg":
        model.add(layers.Dense(1, name="predictions", activation="linear"))

    return model


def mobnet_v3(kind="reg", weights=None) -> Sequential:
    """https://keras.io/api/applications/mobilenet_v3/#mobilenetv3small-function"""

    model_base = MobileNetV3Small(
        include_top=False,
        input_shape=(224, 224, 3),
        weights=weights,
        include_preprocessing=True,
    )
    if weights is not None:
        model_base.trainable = False

    model = rebuild_top(model_base, kind=kind)
    return model


def resnet152_v2(kind="reg", weights=None) -> Sequential:
    """https://keras.io/api/applications/resnet/#resnet152v2-function"""

    model_base = ResNet152V2(
        include_top=False,
        input_shape=(224, 224, 3),
        weights=weights,
    )
    if weights is not None:
        model_base.trainable = False

    model = rebuild_top(model_base, kind=kind)

    # Add preprocessing layer
    preprocess = layers.Lambda(keras.applications.resnet_v2.preprocess_input)
    model = Sequential([preprocess, model])
    model.build()

    return model


def effnet_b0(kind="reg", weights=None) -> Sequential:
    """https://keras.io/api/applications/efficientnet_v2/#efficientnetv2s-function"""

    model_base = EfficientNetB0(
        include_top=False,
        input_shape=(224, 224, 3),
        weights=weights,
        include_preprocessing=True,
    )
    if weights is not None:
        model_base.trainable = False

    model = rebuild_top(model_base, kind=kind)
    return model


def effnet_v2_b0(kind="reg", weights=None) -> Sequential:
    """https://keras.io/api/applications/efficientnet_v2/#efficientnetv2s-function"""

    model_base = EfficientNetV2B0(
        include_top=False,
        input_shape=(224, 224, 3),
        weights=weights,
        include_preprocessing=True,
    )
    if weights is not None:
        model_base.trainable = False

    model = rebuild_top(model_base, kind=kind)
    return model


def effnet_v2_s(kind="reg", weights=None) -> Sequential:
    """https://keras.io/api/applications/efficientnet_v2/#efficientnetv2s-function"""

    model_base = EfficientNetV2S(
        include_top=False,
        input_shape=(224, 224, 3),
        weights=weights,
        include_preprocessing=True,
    )
    if weights is not None:
        model_base.trainable = False

    model = rebuild_top(model_base, kind=kind)
    return model


def effnet_v2_l(kind="reg", weights=None) -> Sequential:
    """https://keras.io/api/applications/efficientnet_v2/#efficientnetv2s-function"""

    model_base = EfficientNetV2L(
        include_top=False,
        input_shape=(224, 224, 3),
        weights=weights,
        include_preprocessing=True,
    )
    if weights is not None:
        model_base.trainable = False

    model = rebuild_top(model_base, kind=kind)
    return model


def small_cnn(image_size, resizing_size=200) -> Sequential:
    """layer normalization entre cada capa y su activación. Batch norm no funca
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

    model.add(layers.Conv2D(32, (3, 3), activation="linear", input_shape=(resizing_size, resizing_size, 4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation="linear"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation="linear"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="linear"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))

    # model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="linear"))

    return model
