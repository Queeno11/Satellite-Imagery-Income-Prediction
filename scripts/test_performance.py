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
# path_programas  = globales[7]
###############################################

import prediction_tools
import custom_models

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


def 