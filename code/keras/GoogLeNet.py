from __future__ import print_function
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, BatchNormalization, concatenate, Input
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D


def Conv2d_BN(x,nb_filter,kernel_size,padding='same',strides=(1,1),name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)
    x = BatchNormalization(axis=3,name=bn_name)(x)
    return x

def Inception(x,nb_filter):
    branch1x1 = Conv2d_BN(x,nb_filter,(1,1),padding='same',strides=(1,1),name=None)

    branch3x3 = Conv2d_BN(x,nb_filter,(1,1),padding='same',strides=(1,1),name=None)
    branch3x3 = Conv2d_BN(branch3x3,nb_filter,(3,3),padding='same',strides=(1,1),name=None)

    branch5x5 = Conv2d_BN(x,nb_filter,(1,1),padding='same',strides=(1,1),name=None)
    branch5x5 = Conv2d_BN(branch5x5,nb_filter,(5,5),padding='same',strides=(1,1),name=None)

    branchpool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same')(x)
    branchpool = Conv2d_BN(branchpool,nb_filter,(1,1),padding='same',strides=(1,1),name=None)

    x = concatenate([branch1x1,branch3x3,branch5x5,branchpool],axis=3)

    return x

def SimpleGoogLeNet(input_shape=(28,28,1),output_size=10):
    inpt = Input(shape=input_shape)
    #padding = 'same'，填充为(步长-1）/2,还可以用ZeroPadding2D((3,3))
    x = Conv2d_BN(inpt,8,(4,4),strides=(1,1),padding='same')
    x = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same')(x)
    x = Conv2d_BN(x,16,(3,3),strides=(1,1),padding='same')
    x = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same')(x)
    x = Inception(x,8)
    '''
    x = Inception(x,20)
    x = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same')(x)
    x = Inception(x,32)
    x = Inception(x,64)
    '''
    x = AveragePooling2D(pool_size=(3,3),strides=(2,2))(x)
    x = Flatten()(x)
    x = Dense(100,activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(output_size,activation='softmax')(x)
    model = Model(inpt,x,name='inception')
    return model

def GoogLeNet(input_shape=(224,224,3),output_size=1000):
    inpt = Input(shape=input_shape)
    #padding = 'same'，填充为(步长-1）/2,还可以用ZeroPadding2D((3,3))
    x = Conv2d_BN(inpt,64,(7,7),strides=(2,2),padding='same')
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
    x = Conv2d_BN(x,192,(3,3),strides=(1,1),padding='same')
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
    x = Inception(x,64)#256
    x = Inception(x,120)#480
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
    x = Inception(x,128)#512
    x = Inception(x,128)
    x = Inception(x,128)
    x = Inception(x,132)#528
    x = Inception(x,208)#832
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
    x = Inception(x,208)
    x = Inception(x,256)#1024
    x = AveragePooling2D(pool_size=(7,7),strides=(7,7),padding='same')(x)
    x = Dropout(0.4)(x)
    x = Dense(1000,activation='relu')(x)
    x = Dense(output_size,activation='softmax')(x)
    model = Model(inpt,x,name='inception')
    return model