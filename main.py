import os

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

RawStockDataDirectory = 'StockData/Raw/'
ProcessedStockDataDirectory = 'StockData/Processed/'


allProcessedData = list()

for stockData in os.listdir(ProcessedStockDataDirectory):
    allProcessedData.append(pd.read_csv(ProcessedStockDataDirectory + stockData, index_col=0))

stock = allProcessedData[0]

scaler = MinMaxScaler()
scaler = scaler.fit(stock)
stockScaled = scaler.transform(stock)

#stock_train, stock_test = train_test_split(stockScaled, test_size=0.2)

n_past = 90
n_future = 14

trainX = []
trainY = []

for i in range(n_past, len(stockScaled) - n_future +1):
    trainX.append(stockScaled[i - n_past:i, 0:stock.shape[1]])
    trainY.append(stockScaled[i:i + n_future, 0])

trainX, trainY = np.array(trainX), np.array(trainY)

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(64, return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(32, activation='relu', return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(32, activation='relu', return_sequences=False))
model.add(tf.keras.layers.Dense(trainY.shape[1]))

model.compile(optimizer='adam', loss='mse')


with tf.device('/GPU:0'):
    history = model.fit(trainX, trainY, epochs=5, batch_size=16, validation_split=0.1)