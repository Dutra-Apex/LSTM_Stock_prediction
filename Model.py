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
#model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics='accuracy')
#Loss function measures how well the model did on training
#Optimizer builds upon the loss function to improve results

#Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)
#Batch size is the total number of training samples presetnt in a single batch
#Epochs are the number of iterations when a entire dataset is passed forward and backward through an LSTM

#Create the test dataset
#Create a new array containing scaled values from index 1543 to 2003

test_data = scaled_data[training_data_len - 60:, :]

#Create the datasets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
# y test are all the values that we want our model to predict

for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])
  
#Convert data to a numpy array
x_test = np.array(x_test)

# Reshape the data so that it is 3d for LSTM
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#Get the models predicted price values
predictions = model.predict(x_test)
#Unscale the values
predictions = scaler.inverse_transform(predictions)
# We want predictions to contain the same values as our y_test 

#Get the models predicted price values
predictions = model.predict(x_test)
#Unscale the values
predictions = scaler.inverse_transform(predictions)
# We want predictions to contain the same values as our y_test 

#Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('Close price')
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Train','Val','Predictions'],loc='lower right')
plt.show()


