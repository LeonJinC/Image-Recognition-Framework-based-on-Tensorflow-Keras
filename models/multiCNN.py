# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 16:51:34 2020

@author: Leon_PC
"""
from keras.layers import Input, Dense, ZeroPadding2D, Dropout
from keras.layers import Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D,Multiply,Reshape,Layer
from keras.models import Model
import tensorflow as tf
import tensorflow.contrib as contrib
import keras.backend as K
import numpy as np
from keras.initializers import RandomUniform
from typing import List

        
class CrossStitchBlock(Layer):
    """
    Cross-stitch block

    References
    ----------
    [1] Cross-stitch Networks for Multi-task Learning, (2017)
    Ishan Misra et al
    """

    def build(self,
              batch_input_shape: List[tf.TensorShape]
              ) -> None:
        stitch_shape = len(batch_input_shape)

        # initialize using random uniform distribution as suggested in ection 5.1
        self.cross_stitch_kernel = self.add_weight(shape=(stitch_shape, stitch_shape),
                                                   initializer=RandomUniform(0., 1.),
                                                   trainable=True,
                                                   name="cross_stitch_kernel")

        # normalize, so that each row will be convex linear combination,
        # here we follow recommendation in paper ( see section 5.1 )
        normalizer = tf.reduce_sum(self.cross_stitch_kernel,
                                   keepdims=True,
                                   axis=0)
        self.cross_stitch_kernel.assign(self.cross_stitch_kernel / normalizer)

    def call(self, input1):
        """
        Forward pass through cross-stitch block

        Parameters
        ----------
        inputs: np.array or tf.Tensor
            List of task specific tensors
        """
        # vectorized cross-stitch unit operation
        input1_reshaped = Flatten(name='input1_reshaped')(input1[0])
        input2_reshaped = Flatten(name='input2_reshaped')(input1[1])
        inputs=[input1_reshaped,input2_reshaped]
        
        x = tf.concat([tf.expand_dims(e, axis=-1) for e in inputs], axis=-1)

        B_ = tf.tile(self.cross_stitch_kernel, [tf.shape(x)[0], 1])  
        self.cross_stitch_kernel = tf.reshape(B_, [tf.shape(x)[0], tf.shape(self.cross_stitch_kernel)[0], tf.shape(self.cross_stitch_kernel)[1]])  
        stitched_output = tf.matmul(x, self.cross_stitch_kernel)

        # split result into tensors corresponding to specific tasks and return
        # Note on implementation: applying tf.squeeze(*) on tensor of shape (None,x,1)
        # produces tensor of shape unknown, so we just extract 0-th element on last axis
        # through gather function
        outputs = [tf.gather(e, 0, axis=-1) for e in tf.split(stitched_output, len(inputs), axis=-1)]

        
        return outputs
    
def multiCNN(input_shape=(200,12,1), classes=49): 
    X_input = Input(input_shape)
    
    # 200，12，12 → 20，12，32
    XA = Conv2D(filters=32, kernel_size=(20,3), strides=(1,1),padding='same',activation='relu', name='conv1')(X_input)
    XA = MaxPooling2D((10,1), strides=(10,1), name='pool1')(XA)
    
    XB = Conv2D(filters=32, kernel_size=(20,3), strides=(1,1),padding='same',activation='relu', name='conv1')(X_input)
    XB = MaxPooling2D((10,1), strides=(10,1), name='pool1')(XB)
    
#    print(XA.shape)

#    XA_1 = Flatten(name='XA_1')(XA)
#    XB_1 = Flatten(name='XA_1')(XB)
#
#    layer_mid=[XA_1,XB_1]
    cs_block = CrossStitchBlock()([XA,XB])
#    
#
##    layer_2 = [Reshape(XA.shape)(stitch) for stitch in cs_block]
#    for stitch in cs_block:
#        print(stitch.shape)
##        stitch=K.reshape(XA.shape,stitch) 
##        print(stitch)
    
    
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