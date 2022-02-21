import tensorflow
from tensorflow.python.client import device_lib

from sklearn.preprocessing import StandardScaler

import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split



def check_gpu():
    print(device_lib.list_local_devices())



def scaling(data):
  scaler = StandardScaler()
  scaler.fit(data)
  return scaler.transform(data)