from __future__ import print_function
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D



def LeNet(input_shape, output_size):
    model = Sequential()
    model.add(Conv2D(32, (5,5), strides=(1,1), input_shape=input_shape,
              padding='valid', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (5,5), strides=(1,1), padding='valid',
              activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(output_size, activation='softmax'))
    return model