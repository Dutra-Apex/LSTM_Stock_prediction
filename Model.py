# Build the LSTM Model

model = Sequential()
# Model with 50 neurons, return_sequences = True because we are using another LSTM layer
# Input shape takes the number of time steps(train.shape[1]) and number of features (1)
# Number of features is simply the closing price
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
