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
  if i <= 61:
    print(x_train)
    print(y_train)

# xtrain contain 60 values, which are use for prediction
# y train contain the first value after 60, which is the value that we want to predict
