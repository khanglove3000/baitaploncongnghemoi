
import numpy as np
import pandas as pd

from .data import create_dataframe
from .layout import html_layout

# machine learning
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler

# deep learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed, GRU

import os.path
from os import path
def create_dataset(X, y, timestep=1):
    Xs, ys = [], []
    for i in range(len(X) - timestep):
        v = X.iloc[i:(i + timestep)].values
        Xs.append(v)        
        ys.append(y.iloc[i + timestep])
    return np.array(Xs), np.array(ys)
    
# Load DataFrame
def detetech_anomalies(train, test, THRESHOLD, time_steps):

    scaler = StandardScaler()
    scaler = scaler.fit(train[['volume']])
    train['volume'] = scaler.transform(train[['volume']])
    test['volume'] = scaler.transform(test[['volume']])

    X_train, y_train = create_dataset(train[['volume']], train.volume, time_steps)
    X_test, y_test = create_dataset(test[['volume']], test.volume, time_steps)
   
    timesteps = X_train.shape[1]
    num_features = X_train.shape[2] 
  

    model = Sequential([
        LSTM(128, input_shape=(timesteps, num_features)),
        Dropout(0.2),
        RepeatVector(timesteps),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        TimeDistributed(Dense(num_features))                 
    ])


    model.compile(loss='mae', optimizer='adam')
    model.summary()

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')
    
    if path.exists('./models/modelanomal.h5'):
        model.load_weights('./models/modelanomal.h5')
        print('load model')
    else:
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.1,
            callbacks = [es],
            shuffle=False
        ) 
        model.save('./models/modelanomal.h5')
    train_mae_loss = model.evaluate(X_test, y_test)
    X_test_pred = model.predict(X_test)
    test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)

 
    test_score_df = pd.DataFrame(test[time_steps:])
    test_score_df['loss'] = test_mae_loss
    test_score_df['threshold'] = THRESHOLD
    test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
    test_score_df['volume'] = test[time_steps:].volume

    anomalies = test_score_df[test_score_df.anomaly == True]

    # bat thuong
    batthuong = pd.DataFrame()
    batthuong[['date','volume','anomaly']] = anomalies[['date','volume','anomaly']]
    volume1 = scaler.inverse_transform(anomalies.volume.values.reshape(-1,1))
    batthuong['volume'] = volume1

    # du lieu test
    test_test = pd.DataFrame()
    volume2 = scaler.inverse_transform(test.volume.values.reshape(-1,1))
    test_test[['date']] = test[['date']]
    test_test['volume'] = volume2
    return test_test, batthuong, test_mae_loss, test_score_df
