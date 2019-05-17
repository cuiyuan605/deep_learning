import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, BatchNormalization, concatenate, Input
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Cropping1D
import numpy as np

def TestModel(input_shape=(10,1)):
    x = Input(shape=input_shape)
    y = Cropping1D(cropping=(1, 5))(x)
    z = Cropping1D(cropping=(5, 1))(x)
    output = concatenate([z,y],axis=2)
    model = Model(inputs=x, outputs=output, name='test')
    return model

if __name__=="__main__":
    print("start!")
    input_data = np.array([[[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]],
                           [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]])
    print("shape:", input_data.shape)
    test = TestModel()
    ret = test.predict(input_data)
    print(ret)