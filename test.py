#Python libraries
import os
import pandas as pd
import numpy as np
import math

import numpy as np
import tensorflow as tf
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, RepeatVector, TimeDistributed
from tensorflow.keras.layers import LSTM, GRU
from tensorflow import keras

# pip install torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from datetime import date, timedelta
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


from package.predict import Normalizer, prepare_data, PredictTrainingData, PredictTestData, PrepareDataPredict, PredictUnseenData
from package.predict import TimeSeriesDataset, LSTModel

# Data preparation
data = pd.read_csv("data\HPG.csv")
print(data)
data_date = data.date
data_close_price = data.close.values
num_data_points = len(data.date)
data_date = list(data_date)

# normalize
scaler = Normalizer()
normalized_data_close_price = scaler.fit_transform(data_close_price)
split_index, data_x_train, data_y_train, data_x_val, data_y_val, data_x_unseen = prepare_data(normalized_data_close_price, num_data_points, scaler, data_date)

dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
dataset_val = TimeSeriesDataset(data_x_val, data_y_val)

print("Train data shape", dataset_train.x.shape, dataset_train.y.shape)
print("Validation data shape", dataset_val.x.shape, dataset_val.y.shape)
model = LSTModel(dataset_train)

predicted_train =  PredictTrainingData(dataset_train, model)
predicted_val =  PredictTestData(dataset_val, model)
PrepareDataPredict(num_data_points, split_index, scaler, predicted_train, predicted_val, data_date, data_close_price)
PredictUnseenData(data_x_unseen, model, scaler, data_y_val, predicted_val, data_date )
