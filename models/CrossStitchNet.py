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



class CrossStitchBlock1(Layer):
    """
    Cross-stitch block

    References
    ----------
    [1] Cross-stitch Networks for Multi-task Learning, (2017)
    Ishan Misra et al
    """
    
    def build(self,batch_input_shape):
        self.stitch_shape=batch_input_shape[0][-1]
        self.cross_stitch_kernel = self.add_weight(shape=(self.stitch_shape*2, self.stitch_shape*2),
                                                   initializer=RandomUniform(0., 1.),
                                                   trainable=True,
                                                   name="cross_stitch_kernel")
        
        # normalize, so that each row will be convex linear combination,
        # here we follow recommendation in paper ( see section 5.1 )
        normalizer = tf.reduce_sum(self.cross_stitch_kernel,
                                   keepdims=True,
                                   axis=0)
        self.cross_stitch_kernel.assign(self.cross_stitch_kernel / normalizer)

    def call(self, inputs):
        """
        Forward pass through cross-stitch block

        Parameters
        ----------
        inputs: np.array or tf.Tensor
            List of task specific tensors
        """

        input = tf.concat((inputs[0],inputs[1]), axis=-1)

        output = tf.matmul(input, self.cross_stitch_kernel)
        input1_shape = list(-1 if s.value is None else s.value for s in inputs[0].shape)
        input2_shape = list(-1 if s.value is None else s.value for s in inputs[1].shape)
        output1 = tf.reshape(output[:, :input1_shape[1]], shape=input1_shape)
        output2 = tf.reshape(output[:, input1_shape[1]:], shape=input2_shape)
        
        return [output1,output2]  
    
def cross_stritch_layer1(i,input1,input2):
    XA_in = Flatten(name='cross_flatten_1_'+str(i)+'')(input1)
    XA_in= Dense(K.int_shape(input1)[1]*K.int_shape(input1)[2]*K.int_shape(input1)[3],name='cross_dense_1_'+str(i))(XA_in)
    XB_in = Flatten(name='cross_flatten_2_'+str(i)+'')(input2)
    XB_in= Dense(K.int_shape(input2)[1]*K.int_shape(input2)[2]*K.int_shape(input2)[3],name='cross_dense_2_'+str(i))(XB_in)

    cs_block = CrossStitchBlock1()([XA_in,XB_in])
    XA_out=cs_block[0]
    XB_out=cs_block[1] 
    
    XA_out = Reshape((K.int_shape(input1)[1],K.int_shape(input1)[2],K.int_shape(input1)[3]))(XA_out)
    XB_out = Reshape((K.int_shape(input2)[1],K.int_shape(input2)[2],K.int_shape(input2)[3]))(XB_out)
    return [XA_out,XB_out]


class CrossStitchBlock2(Layer):
    """
    Cross-stitch block

    References
    ----------
    [1] Cross-stitch Networks for Multi-task Learning, (2017)
    Ishan Misra et al
    """

    def build(self,batch_input_shape: List[tf.TensorShape]) -> None:
        stitch_shape = len(batch_input_shape)
#        print(stitch_shape)
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

    def call(self, inputs):
        """
        Forward pass through cross-stitch block

        Parameters
        ----------
        inputs: np.array or tf.Tensor
            List of task specific tensors
        """
        # vectorized cross-stitch unit operation
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

def cross_stritch_layer2(i,input1,input2):
    XA_in = Flatten(name='cross_flatten_1_'+str(i)+'')(input1)
    XA_in= Dense(K.int_shape(input1)[1]*K.int_shape(input1)[2]*K.int_shape(input1)[3],name='cross_dense_1_'+str(i))(XA_in)
    XB_in = Flatten(name='cross_flatten_2_'+str(i)+'')(input2)
    XB_in= Dense(K.int_shape(input2)[1]*K.int_shape(input2)[2]*K.int_shape(input2)[3],name='cross_dense_2_'+str(i))(XB_in)

    cs_block = CrossStitchBlock2()([XA_in,XB_in])
    XA_out=cs_block[0]
    XB_out=cs_block[1] 
#    print(XA_out)
    XA_out = Reshape((K.int_shape(input1)[1],K.int_shape(input1)[2],K.int_shape(input1)[3]))(XA_out)
    XB_out = Reshape((K.int_shape(input2)[1],K.int_shape(input2)[2],K.int_shape(input2)[3]))(XB_out)
    return [XA_out,XB_out]

def CrossStitchNet(input_shape=(100,120,1), classes=49): 
    X_input = Input(input_shape)
    
    X= Conv2D(filters=1, kernel_size=(3,3), strides=(2,2),padding='same', activation='relu',name='conv1')(X_input)
    X = MaxPooling2D((2,2), strides=(1,1), name='pool2')(X)

    XA = Conv2D(filters=2, kernel_size=(5,5), strides=(2,2),padding='same',activation='relu', name='conv2_1')(X)
    XA = MaxPooling2D((2,2), strides=(2,2), name='pool1_1')(XA)
    
    
    XB = Conv2D(filters=2, kernel_size=(5,5), strides=(2,2),padding='same',activation='relu', name='conv2_2')(X)
    XB = MaxPooling2D((2,2), strides=(2,2), name='pool1_2')(XB)

#    XA,XB=cross_stritch_layer1(1,XA,XB)
#    XA = Conv2D(filters=2, kernel_size=(3,3), strides=(1,1),padding='same',activation='relu', name='conv3_1')(XA)
#    XB = Conv2D(filters=2, kernel_size=(3,3), strides=(1,1),padding='same',activation='relu', name='conv3_2')(XB)
#    X=add([XA,XB])
    
    XA,XB=cross_stritch_layer2(1,XA,XB)
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