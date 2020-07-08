# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 11:33:46 2020

@author: Santanu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras as kr

train_dataframe = pd.read_csv("fashion-mnist_train.csv") 
test_dataframe = pd.read_csv("fashion-mnist_test.csv") 

print(train_dataframe.head())

##Convert DataFrames to nparray
train_data = np.array(train_dataframe, dtype = "float32")
test_data = np.array(test_dataframe, dtype = "float32")


##pixel values and labels separate labels as y and pixels as x
##traindata and normalize
x_train_data = train_data[:,1:]/255
y_train_data = train_data[:,0]
##testData and normalize
x_test_data = test_data[:,1:]/255
y_test_data = test_data[:,0]

data = x_train_data[50].reshape((28,28))
plt.imshow(data,cmap="gray")
plt.show()

x_train_data = x_train_data.reshape(x_train_data.shape[0], 28,28,1) 
x_test_data = x_test_data.reshape(x_test_data.shape[0], 28,28,1) 

print(x_train_data.shape)
print(x_test_data.shape)

model = kr.models.Sequential()
model.add(kr.layers.Convolution2D(32, (3,3), activation="relu",  input_shape =(28,28,1)))
model.add(kr.layers.Convolution2D(64, (3,3), activation="relu"))
model.add(kr.layers.Dropout(0.25))
model.add(kr.layers.MaxPooling2D(2,2))

model.add(kr.layers.Convolution2D(32, (5,5), activation="relu"))
model.add(kr.layers.Convolution2D(64, (5,5), activation="relu"))

model.add(kr.layers.Flatten())
model.add(kr.layers.Dense(100, activation="softmax"))
model.add(kr.layers.Dense(10, activation="softmax"))


model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics = ["accuracy"])

hist = model.fit(x_train_data, y_train_data, epochs =80, batch_size=256, validation_split = 0.2, shuffle =True)


plt.plot(hist.history["acc"], c="red")
plt.plot(hist.history["loss"])
plt.plot(hist.history["val_acc"], c="green")
plt.plot(hist.history["val_loss"], c="black")

plt.show()





