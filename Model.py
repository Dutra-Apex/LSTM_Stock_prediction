import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# Build the LSTM Model

model = Sequential()
# Model with 50 neurons, return_sequences = True because we are using another LSTM layer
# Input shape takes the number of time steps(train.shape[1]) and number of features (1)
# Number of features is simply the closing price
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))


# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics='accuracy')
#Loss function measures how well the model did on training
#Optimizer builds upon the loss function to improve results
