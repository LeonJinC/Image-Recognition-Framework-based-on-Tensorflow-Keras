# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 16:51:34 2020

@author: Leon_PC
"""
from keras.layers import Input, Dense, ZeroPadding2D, Dropout, Activation, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from keras.models import Model
def CNN(input_shape=(200,12,1), classes=49): 
    X_input = Input(input_shape)
    
    # 200，12，12 → 20，12，32
    X = Conv2D(filters=32, kernel_size=(20,3), strides=(1,1),padding='same',activation='relu', name='conv1')(X_input)
    X = MaxPooling2D((10,1), strides=(10,1), name='pool1')(X)
    
    # 20，12，32 → 6，6，64
    X = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1),padding='same', activation='relu',name='conv2')(X)
    X = MaxPooling2D((3,2), strides=(3,2), name='pool2')(X)
    
    # 6，6，64 → 3，3，128
    X = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1),padding='same', activation='relu',name='conv3')(X)
    X = MaxPooling2D((2,2), strides=(2,2), name='pool3')(X)
    
    # 3*3*128 = 1152
    X = Flatten(name='flatten')(X)
    X = Dropout(0.5)(X) 
    X = Dense(128,activation='relu',name='fc1')(X)
    X = Dropout(0.5)(X)
    X = Dense(classes, activation='softmax', name='fc2')(X)
    
    model = Model(inputs=X_input, outputs=X, name='CNN')
#    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model