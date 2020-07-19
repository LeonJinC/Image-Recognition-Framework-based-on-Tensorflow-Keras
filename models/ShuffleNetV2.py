
# 数据预处理

import h5py
import numpy as np
import tensorflow as tf 
import keras
from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization, Flatten, Conv2D, Reshape
from keras.layers import ZeroPadding2D, AveragePooling2D, MaxPooling2D
from keras.layers import DepthwiseConv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Add, MaxPooling2D
from keras.layers import Multiply, Lambda, Concatenate
from keras import backend as K
from keras.models import Model
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
from utils import struct
import pickle
from tensorflow.keras.callbacks import TensorBoard


tf.reset_default_graph()

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

file = h5py.File('DB2/DB2_S1-30_norm_winlen200_slide200.h5','r')
imageData   = file['trainx'][:]
imageLabel  = file['trainy'][:] 
file.close()

imageData=imageData.reshape(-1,200,12)
imageLabel=imageLabel.astype('int64')
print(imageData.shape)
print(len(imageLabel))

# 随机打乱数据和标签
N = imageData.shape[0]
index = np.random.permutation(N)
data  = imageData[index,:,:]
label = imageLabel[index]

# 对数据升维,标签one-hot
data  = np.expand_dims(data, axis=3)
label = convert_to_one_hot(label,49).T

data=data.repeat(3,axis=3)

# 划分数据集
N = data.shape[0]
num_train = round(N*0.8)
X_train = data[0:num_train,:,:,:]
Y_train = label[0:num_train,:]
X_test  = data[num_train:N,:,:,:]
Y_test  = label[num_train:N,:]

print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

#写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


def channel_split(x, name=''):
    # 输入进来的通道数
    in_channles = x.shape.as_list()[-1]
    ip = in_channles // 2
    # 对通道数进行分割
    c_hat = Lambda(lambda z: z[:, :, :, 0:ip], name='%s/sp%d_slice' % (name, 0))(x)
    c = Lambda(lambda z: z[:, :, :, ip:], name='%s/sp%d_slice' % (name, 1))(x)
    return c_hat, c

def channel_shuffle(x):
    height, width, channels = x.shape.as_list()[1:]
    channels_per_split = channels // 2
    # 通道交换
    x = K.reshape(x, [-1, height, width, 2, channels_per_split])
    x = K.permute_dimensions(x, (0,1,2,4,3))
    x = K.reshape(x, [-1, height, width, channels])
    return x


def shuffle_unit(inputs, out_channels, bottleneck_ratio, strides=2, stage=1, block=1):
    bn_axis = -1

    prefix = 'stage{}/block{}'.format(stage, block)

    # [116, 232, 464]
    bottleneck_channels = int(out_channels * bottleneck_ratio/2)
    if strides < 2:
        c_hat, c = channel_split(inputs, '{}/spl'.format(prefix))
        inputs = c

    # [116, 232, 464]
    x = Conv2D(bottleneck_channels, kernel_size=(1,1), strides=1, padding='same', name='{}/1x1conv_1'.format(prefix))(inputs)
    x = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_1'.format(prefix))(x)
    x = Activation('relu', name='{}/relu_1x1conv_1'.format(prefix))(x)

    # 深度可分离卷积
    x = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same', name='{}/3x3dwconv'.format(prefix))(x)
    x = BatchNormalization(axis=bn_axis, name='{}/bn_3x3dwconv'.format(prefix))(x)
    
    # [116, 232, 464]
    x = Conv2D(bottleneck_channels, kernel_size=1,strides=1,padding='same', name='{}/1x1conv_2'.format(prefix))(x)
    x = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_2'.format(prefix))(x)
    x = Activation('relu', name='{}/relu_1x1conv_2'.format(prefix))(x)

    # 当strides等于2的时候，残差边需要添加卷积
    if strides < 2:
        ret = Concatenate(axis=bn_axis, name='{}/concat_1'.format(prefix))([x, c_hat])
    else:
        s2 = DepthwiseConv2D(kernel_size=3, strides=2, padding='same', name='{}/3x3dwconv_2'.format(prefix))(inputs)
        s2 = BatchNormalization(axis=bn_axis, name='{}/bn_3x3dwconv_2'.format(prefix))(s2)

        s2 = Conv2D(bottleneck_channels, kernel_size=1,strides=1,padding='same', name='{}/1x1_conv_3'.format(prefix))(s2)
        s2 = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_3'.format(prefix))(s2)
        s2 = Activation('relu', name='{}/relu_1x1conv_3'.format(prefix))(s2)
        ret = Concatenate(axis=bn_axis, name='{}/concat_2'.format(prefix))([x, s2])

    ret = Lambda(channel_shuffle, name='{}/channel_shuffle'.format(prefix))(ret)

    return ret


def block(x, channel_map, bottleneck_ratio, repeat=1, stage=1):
    x = shuffle_unit(x, out_channels=channel_map[stage-1],
                      strides=2,bottleneck_ratio=bottleneck_ratio,stage=stage,block=1)

    for i in range(1, repeat+1):
        x = shuffle_unit(x, out_channels=channel_map[stage-1],strides=1,
                          bottleneck_ratio=bottleneck_ratio,stage=stage, block=(1+i))

    return x

def ShuffleNetV2(input_tensor=None,
                 pooling='max',
                 input_shape=(200,12,3),
                 num_shuffle_units=[3,7,3],
                 scale_factor=1,
                 bottleneck_ratio=1,
                 classes=49):
    name = 'ShuffleNetV2_{}_{}_{}'.format(scale_factor, bottleneck_ratio, "".join([str(x) for x in num_shuffle_units]))

    out_dim_stage_two = {0.5:48, 1:116, 1.5:176, 2:244}

    out_channels_in_stage = np.array([1,1,2,4])
    out_channels_in_stage *= out_dim_stage_two[scale_factor]  #  calculate output channels for each stage
    out_channels_in_stage[0] = 24  # first stage has always 24 output channels
    out_channels_in_stage = out_channels_in_stage.astype(int)

    img_input = Input(shape=input_shape)

    x = Conv2D(filters=out_channels_in_stage[0], kernel_size=(3, 3), padding='same', use_bias=False, strides=(2, 2),
               activation='relu', name='conv1')(img_input)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool1')(x)

    for stage in range(len(num_shuffle_units)):
        repeat = num_shuffle_units[stage]
        x = block(x, out_channels_in_stage,
                   repeat=repeat,
                   bottleneck_ratio=bottleneck_ratio,
                   stage=stage + 2)

    if scale_factor!=2:
        x = Conv2D(1024, kernel_size=1, padding='same', strides=1, name='1x1conv5_out', activation='relu')(x)
    else:
        x = Conv2D(2048, kernel_size=1, padding='same', strides=1, name='1x1conv5_out', activation='relu')(x)
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)

    x = Dense(classes, name='fc')(x)
    x = Activation('softmax', name='softmax')(x)

    inputs = img_input

    model = Model(inputs, x, name=name)

    return model


model = ShuffleNetV2(input_shape = (200, 12, 3))
model.summary()

# 训练原始数据

import time
start = time.time()
model_name = "DB2-ShuffleNetV2-{}".format(int(time.time()))
log_dir='logs/{}'.format(model_name)
print(log_dir)
tensorboard = TensorBoard(log_dir=log_dir)


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

history = LossHistory()         # 创建一个history实例

model.fit(X_train, Y_train, epochs=20, batch_size=64, verbose=1,
          validation_data=(X_test, Y_test),callbacks=[history,tensorboard])

preds_train = model.evaluate(X_train, Y_train)
print("Train Loss = " + str(preds_train[0]))
print("Train Accuracy = " + str(preds_train[1]))

preds_test  = model.evaluate(X_test, Y_test)
print("Test Loss = " + str(preds_test[0]))
print("Test Accuracy = " + str(preds_test[1]))

end = time.time()
print("time:",end-start)

history.loss_plot('epoch')

import os
os.system('"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" http://localhost:6006')
os.system('"D:\\ProgramData\\Anaconda3\\envs\\tensorflow_gpu\\Scripts\\tensorboard.exe" --logdir='+log_dir)


# history.loss_plot('epoch')
