import math
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')


class LSTMPredictor:
    def __init__(self, ticker, start, end):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.stock_data = yf.download(ticker, start=start, end=end)
        self.scaler, self.scaled_data, self.train_data, self.training_data_len, self.values = self.standardize()
        self.x_train, self.y_train, self.x_test, self.y_test = self.test_train_split()
        self.model = self.model_buildup()
        self.fit()

    def standardize(self):
        close_prices = self.stock_data['Adj Close']
        values = close_prices.values
        training_data_len = math.ceil(len(values) * 0.8)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(values.reshape(-1, 1))
        train_data = scaled_data[0: training_data_len, :]
        return scaler, scaled_data, train_data, training_data_len, values

    def test_train_split(self):
        x_train = []
        y_train = []
        for i in range(60, len(self.train_data)):
            x_train.append(self.train_data[i - 60:i, 0])
            y_train.append(self.train_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        test_data = self.scaled_data[self.training_data_len - 60:, :]
        x_test = []
        y_test = self.values[self.training_data_len:]

        for i in range(60, len(test_data)):
            x_test.append(test_data[i - 60:i, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        return x_train, y_train, x_test, y_test

    def model_buildup(self):
        model = keras.Sequential()
        model.add(layers.LSTM(100, return_sequences=True, input_shape=(self.x_train.shape[1], 1)))
        model.add(layers.LSTM(100, return_sequences=False))
        model.add(layers.Dense(25))
        model.add(layers.Dense(1))
        return model

    def fit(self):
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(self.x_train, self.y_train, batch_size=1, epochs=6)

    def predict(self):
        predictions = self.model.predict(self.x_test)
        predictions = self.scaler.inverse_transform(predictions)
        rmse = np.sqrt(np.mean(predictions - self.y_test) ** 2)
        print(f"loss: {rmse}")
        return predictions, rmse

    def plot(self):
        data = self.stock_data.filter(['Adj Close'])
        train = data[:self.training_data_len]
        validation = data[self.training_data_len:]
        validation['Predictions'] = self.predict()[0]
        plt.figure(figsize=(16, 8))
        plt.title('Model')
        plt.xlabel('Date')
        plt.ylabel('Adj Close Price USD ($)')
        plt.plot(train)
        plt.plot(validation[['Adj Close', 'Predictions']])
        plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
        plt.show()


if __name__ == '__main__':
    l = LSTMPredictor('TSLA', '2018-01-01', '2023-02-16')
    l.plot()






