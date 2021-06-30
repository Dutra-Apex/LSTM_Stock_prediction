#Create the training dataset and the scaled traing dataset
train_data = scaled_data[0:training_data_len, :]

#Split the data into xtrain and ytrain:
#Independant variables
x_train = []

#Dependant variables
y_train = []

for i in range(60, len(train_data)):
  x_train.append(train_data[i-60:i, 0]) 
  y_train.append(train_data[i,0])
  
# xtrain contain 60 values, which are use for prediction
# y train contain the first value after 60, which is the value that we want to predict

#Convert both train sets to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
