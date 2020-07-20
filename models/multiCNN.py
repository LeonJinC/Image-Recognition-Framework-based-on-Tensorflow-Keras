# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 16:51:34 2020

@author: Leon_PC
"""
from keras.layers import Input, Dense, ZeroPadding2D, Dropout
from keras.layers import Activation, BatchNormalization, Flatten, Conv2D,Lambda
from keras.layers import AveragePooling2D, MaxPooling2D,Multiply,Reshape,Layer,add
from keras.models import Model
import tensorflow as tf
import tensorflow.contrib as contrib
import keras.backend as K
import numpy as np
from keras.initializers import RandomUniform
from typing import List
from keras.layers.merge import concatenate


def multiCNN(input_shape=(100,120,1), classes=49): 
    X_input = Input(input_shape)
    
    X= Conv2D(filters=1, kernel_size=(3,3), strides=(2,2),padding='same', activation='relu',name='conv1')(X_input)
    X = MaxPooling2D((2,2), strides=(1,1), name='pool2')(X)

    XA = Conv2D(filters=2, kernel_size=(5,5), strides=(2,2),padding='same',activation='relu', name='conv2_1')(X)
    XA = MaxPooling2D((2,2), strides=(2,2), name='pool1_1')(XA)
    
    
    XB = Conv2D(filters=2, kernel_size=(5,5), strides=(2,2),padding='same',activation='relu', name='conv2_2')(X)
    XB = MaxPooling2D((2,2), strides=(2,2), name='pool1_2')(XB)
    

    XA = Conv2D(filters=2, kernel_size=(3,3), strides=(1,1),padding='same',activation='relu', name='conv3_1')(XA)
    XB = Conv2D(filters=2, kernel_size=(3,3), strides=(1,1),padding='same',activation='relu', name='conv3_2')(XB)
    
    X=add([XA,XB])


##    # 3*3*128 = 1152
    X = Flatten(name='flatten')(X)
    X = Dropout(0.5)(X) 
    X = Dense(128,activation='relu',name='fc1')(X)
    X = Dropout(0.5)(X)
    X = Dense(classes, activation='softmax', name='fc2')(X)
    model = Model(inputs=X_input, outputs=X, name='CNN')

    return model