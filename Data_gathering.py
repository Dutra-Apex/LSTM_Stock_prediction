import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#Pick stock
df = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2021-02-01')
#Show data
df

# Vizualize closing price history
plt.figure(figsize=(16,8))
plt.title('Closing price history')
plt.plot(df['Close'])
plt.ylabel('Close Price')
plt.xlabel('Date')
plt.show()

#Create new dataframe with only close column
data = df.filter(['Close'])

#Converts the dataframe to a numpy array
dataset = data.values
#Get the number of rows to train the model on
# This will train on 80% of the data that we have
training_data_len = math.ceil(len(dataset) * .8)
training_data_len

#Scale the data (good practice to pre process the data before presenting to a Neural Network)
scaler = MinMaxScaler(feature_range=(0,1))
#Computes the min and max values values used on scaler (inclusive)
# and transform the data accordingly
scaled_data = scaler.fit_transform(dataset)
