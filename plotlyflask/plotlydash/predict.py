#Python libraries
import os
import pandas as pd
import numpy as np
import math

import os.path
from os import path

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

  
import os.path
from os import path

config = {
    
    "data": {
        "window_size": 20,
        "train_split_size": 0.80,
    }, 
    "plots": {
        "show_plots": True,
        "xticks_interval": 90,
        "color_actual": "#001f3f",
        "color_train": "#3D9970",
        "color_val": "#0074D9",
        "color_pred_train": "#3D9970",
        "color_pred_val": "#0074D9",
        "color_pred_test": "#FF4136",
    },
    "model": {
        "input_size": 1, # since we are only using 1 feature, close price
        "num_lstm_layers": 2,
        "lstm_size": 32,
        "dropout": 0.2,
    },
    "training": {
        "device": "cpu", # "cuda" or "cpu"
        "batch_size": 64,
        "num_epoch": 100,
        "learning_rate": 0.01,
        "scheduler_step_size": 40,
    }
}


class Normalizer():
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, x):
        self.mu = np.mean(x, axis=(0), keepdims=True)
        self.sd = np.std(x, axis=(0), keepdims=True)
        normalized_x = (x - self.mu)/self.sd
        return normalized_x

    def inverse_transform(self, x):
        return (x*self.sd) + self.mu

def prepare_data_x(x, window_size):
    # perform windowing
    n_row = x.shape[0] - window_size + 1
    output = np.lib.stride_tricks.as_strided(x, shape=(n_row,window_size), strides=(x.strides[0],x.strides[0]))
    return output[:-1], output[-1]

def prepare_data_y(x, window_size):
    # # perform simple moving average
    # output = np.convolve(x, np.ones(window_size), 'valid') / window_size

    # use the next day as label
    output = x[window_size:]
    return output

def prepare_data(normalized_data_close_price, num_data_points, scaler, data_date):
    data_x, data_x_unseen = prepare_data_x(normalized_data_close_price, window_size=config["data"]["window_size"])
    data_y = prepare_data_y(normalized_data_close_price, window_size=config["data"]["window_size"])

    # split dataset

    split_index = int(data_y.shape[0]*config["data"]["train_split_size"])
    data_x_train = data_x[:split_index]
    data_x_val = data_x[split_index:]
    data_y_train = data_y[:split_index]
    data_y_val = data_y[split_index:]

    
    # prepare data for plotting

    to_plot_data_y_train = np.zeros(num_data_points)
    to_plot_data_y_val = np.zeros(num_data_points)

    to_plot_data_y_train[config["data"]["window_size"]:split_index+config["data"]["window_size"]] = scaler.inverse_transform(data_y_train)
    to_plot_data_y_val[split_index+config["data"]["window_size"]:] = scaler.inverse_transform(data_y_val)

    to_plot_data_y_train = np.where(to_plot_data_y_train == 0, None, to_plot_data_y_train)
    to_plot_data_y_val = np.where(to_plot_data_y_val == 0, None, to_plot_data_y_val)

    ## plots    

    return split_index, data_x_train, data_y_train, data_x_val, data_y_val, data_x_unseen

class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        x = np.expand_dims(x, 2) # in our case, we have only 1 feature, so we need to convert `x` into [batch, sequence, features] for LSTM
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])


def LSTModel(dataset_train):
    # xay dung model LSTM
    model = Sequential()
    model.add(LSTM(units=50, return_sequences = True, input_shape=(dataset_train.x.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=32))
    # model.add(Dense(units=16))
    model.add(Dense(units=1))
    model.compile(optimizer = 'adam', loss='mean_squared_error', metrics=['acc'])


    if path.exists('./models/model_predict_LSTM.h5'):
        model.load_weights('./models/model_predict_LSTM.h5')
    else:
        model.fit(dataset_train.x, dataset_train.y, epochs=100, batch_size=32, validation_split=0.2)
        model.save('./models/model_predict_LSTM.h5')
    return model

# here we re-initialize dataloader so the data doesn't shuffled, so we can plot the values by date

# predict on the training data, to see how well the model managed to learn and memorize

def PredictTrainingData(dataset_train, model):
    train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=False)
    predicted_train = np.array([])
    for idx, (x, y) in enumerate(train_dataloader):
        x = x.to(config["training"]["device"])
        x = np.array(x, dtype='float')
        out = model(x)
        out = out.cpu().numpy().reshape(-1)
        predicted_train = np.concatenate((predicted_train, out))
    return predicted_train

# predict on the validation data, to see how the model does
def PredictTestData(dataset_val, model):
    val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=False)
    predicted_val = np.array([])
    for idx, (x, y) in enumerate(val_dataloader):
        x = x.to(config["training"]["device"])
        x = np.array(x, dtype='float')
        out = model(x)
        out = out.cpu().numpy().reshape(-1)
        predicted_val = np.concatenate((predicted_val, out))
    return predicted_val

def PrepareDataPredict(num_data_points, split_index, scaler, predicted_train, predicted_val, data_date, data_close_price):
    # prepare data for plotting, show predicted prices

    to_plot_data_y_train_pred = np.zeros(num_data_points)
    to_plot_data_y_val_pred = np.zeros(num_data_points)

    to_plot_data_y_train_pred[config["data"]["window_size"]:split_index+config["data"]["window_size"]] = scaler.inverse_transform(predicted_train)
    to_plot_data_y_val_pred[split_index+config["data"]["window_size"]:] = scaler.inverse_transform(predicted_val)

    to_plot_data_y_train_pred = np.where(to_plot_data_y_train_pred == 0, None, to_plot_data_y_train_pred)
    to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)


    # plots

    return data_close_price, to_plot_data_y_train_pred, to_plot_data_y_val_pred


def PrepareZoomValidation(split_index, scaler, predicted_val, data_date, data_y_val):
    # prepare data for plotting, zoom in validation

    to_plot_data_y_val_subset = scaler.inverse_transform(data_y_val)
    to_plot_predicted_val = scaler.inverse_transform(predicted_val)
    to_plot_data_date = data_date[split_index+config["data"]["window_size"]:]
 

    # plots

    return to_plot_data_y_val_subset, to_plot_predicted_val, to_plot_data_date

def PredictUnseenData(data_x_unseen, model, scaler, data_y_val, predicted_val, data_date ):
    x = torch.tensor(data_x_unseen).float().to(config["training"]["device"]).unsqueeze(0).unsqueeze(2) # this is the data type and shape required, [batch, sequence, feature]
    x = np.array(x, dtype='float')
    prediction = model(x)
    prediction = prediction.cpu().numpy()
    prediction = scaler.inverse_transform(prediction)[0]

    plot_range = 10
    to_plot_data_y_val = np.zeros(plot_range)
    to_plot_data_y_val_pred = np.zeros(plot_range)
    to_plot_data_y_test_pred = np.zeros(plot_range)

    to_plot_data_y_val[:plot_range-1] = scaler.inverse_transform(data_y_val)[-plot_range+1:]
    to_plot_data_y_val_pred[:plot_range-1] = scaler.inverse_transform(predicted_val)[-plot_range+1:]

    to_plot_data_y_test_pred[plot_range-1] = prediction

    to_plot_data_y_val = np.where(to_plot_data_y_val == 0, None, to_plot_data_y_val)
    to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)
    to_plot_data_y_test_pred = np.where(to_plot_data_y_test_pred == 0, None, to_plot_data_y_test_pred)

    # # plot
    plot_date_test = data_date[-plot_range+1:]

    from datetime import datetime, timedelta
    end_day =  plot_date_test[-1]
    end_day = end_day.to_pydatetime()
    end_day = datetime.strptime(str(end_day), '%Y-%m-%d %H:%M:%S')
    next_trading_day = str(end_day+timedelta(days=1))
    plot_date_test.append(next_trading_day)
    
    return plot_date_test, to_plot_data_y_val, to_plot_data_y_val_pred, to_plot_data_y_test_pred, prediction
    

    