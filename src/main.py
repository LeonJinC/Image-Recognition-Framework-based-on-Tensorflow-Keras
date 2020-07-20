import h5py
import numpy as np
import tensorflow as tf 
from keras import backend as K
import keras
from keras.layers import Input, Dense, ZeroPadding2D, Dropout, Activation, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from keras.models import Model
import matplotlib.pyplot as plt

from tqdm import tqdm
import logging#类似于print，用于输出运行日志
import os
from datetime import datetime
import pickle
from tensorflow.keras.callbacks import TensorBoard
import time
import os,pathlib
from models.mobilenets import MobileNet
from models.CNN import CNN
from models.multiCNN import multiCNN
from models.CrossStitchNet import CrossStitchNet

K.clear_session()
tf.reset_default_graph()

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

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


def myprint(s):
    with open('main.log','a') as f:
        print(s, file=f)



        
class ImRecognition():
    def __init__(self,istrain='train'):
        self.epochs=20
        self.batch_size=128
        self.verbose=1
        self.classes = 49
        self.input_shape = (200, 36, 1)
        self.data_time=datetime.now()
        self.data_path='../DB2/DB2_S1-1_norm_winlen200_slide200.h5'
        
        logging.basicConfig(level=logging.DEBUG, format = "[%(asctime)s] %(message)s", datefmt="%m%d %H:%M:%S")
        self.logger = logging.getLogger(str(self.data_time))
        self.logger.addHandler(logging.FileHandler('./main.log'))
        
        self.logger.info("\n\n")
        self.logger.info("*********************************************************************")
        self.logger.info("*********************************************************************")
        self.logger.info("**************Exp data time: "+str(self.data_time)+"**************")
        self.logger.info("*********************************************************************")
        self.logger.info("*********************************************************************")
        

        self.model = self.build_model()
             
        if istrain=='train':
            self.logger.info("Train config setting ...")
            self.logger.info("epochs: "+str(self.epochs))
            self.logger.info("batch_size: "+str(self.batch_size))
            self.logger.info("verbose: "+str(self.verbose))
        
            self.getData()
            os.makedirs(self.checkpoint_dir)
            self.logger.info("saving train_data: "+self.checkpoint_dir+'/traindata.h5')
            file = h5py.File(self.checkpoint_dir+'/traindata.h5','w')  
            file.create_dataset('trainx', data = self.X_train)  
            file.create_dataset('trainy', data = self.Y_train)  
            file.close() 
            
            self.logger.info("saving test_data: "+self.checkpoint_dir+'/testdata.h5')
            file = h5py.File(self.checkpoint_dir+'/testdata.h5','w')  
            file.create_dataset('testx', data = self.X_test)  
            file.create_dataset('testy', data = self.Y_test)  
            file.close()          
        
    def build_model(self):
        self.logger.info("build model ...")
                
#        model = CNN(input_shape = self.input_shape,classes = self.classes)
#        model = multiCNN(input_shape = self.input_shape,classes = self.classes)
        model = MobileNet(input_shape = self.input_shape,classes = self.classes,attention_module = 'cbam_block')
#        model = MobileNet(input_shape = self.input_shape,classes = self.classes)
#        model = CrossStitchNet(input_shape = self.input_shape,classes = self.classes)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        model.summary(print_fn=myprint)
        self.logger.info("\n")

        # 创建一个history实例
        self.history = LossHistory() 

        # 创建一个tensorboard实例
        model_name = "DB2-Conv2D-{0:%Y-%m-%dT%H-%M-%S/}".format(self.data_time)
        self.log_dir='logs/{}'.format(model_name)
        self.logger.info("log_dir: "+self.log_dir)
        self.tensorboard = TensorBoard(log_dir=self.log_dir)

        # 创建一个ModelCheckpoint实例
        self.checkpoint_path = self.log_dir+"/cp-{epoch:04d}.ckpt"  #全路径
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path) #所在文件路径
        self.logger.info("checkpoint_dir: "+self.checkpoint_dir)
        # Create checkpoint callback
        self.cp_callback = keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                                      save_weights_only=True,
                                                      verbose=1,
                                                      period=5)# Save weights, every 5-epochs.
        self.logger.info("\n")

        return model
    
    def getData(self):
        self.logger.info("\n")
        self.logger.info("dataset load ...")
        self.logger.info("data_path: "+str(self.data_path))
        self.logger.info("input_shape: "+str(self.input_shape))
        self.logger.info("classes: "+str(self.classes))
        
        file = h5py.File(self.data_path,'r')
        imageData   = file['trainx'][:]
        imageLabel  = file['trainy'][:] 
        file.close()
        repeat1=1
        repeat2=1
        if self.input_shape[0]*self.input_shape[1]!=(200*12):
            
            assert self.input_shape[0]/200>=1,'wrong input_shape[0]'
            if self.input_shape[0]/200>=1:
                repeat1=self.input_shape[0]/200
                
            assert self.input_shape[1]/12>=1,'wrong input_shape[1]'
            if self.input_shape[1]/12>=1:
                repeat2=self.input_shape[1]/12
                
            imageData=imageData.reshape(-1,int(self.input_shape[0]/repeat1),int(self.input_shape[1]/repeat2))
        else:
            imageData=imageData.reshape(-1,int(repeat1*self.input_shape[0]),int(repeat2*self.input_shape[1]))
        
        
        imageLabel=imageLabel.astype('int64')
        
        # 随机打乱数据和标签
        N = imageData.shape[0]
        index = np.random.permutation(N)
        data  = imageData[index,:,:]
        label = imageLabel[index]
        
    
        # 对数据升维,标签one-hot
        data  = np.expand_dims(data, axis=3)#(?, 200, 12)->(?, 200, 12, 1)
        if repeat1>1:
            data = data.repeat(repeat1,axis=1)
        if repeat2>1:
            data = data.repeat(repeat2,axis=2)

        label = convert_to_one_hot(label,self.classes).T
        
        # 划分数据集
        N = data.shape[0]
        num_train = round(N*0.8)
        
        self.X_train = data[0:num_train,:,:,:]
        self.Y_train = label[0:num_train,:]
        self.X_test  = data[num_train:N,:,:,:]
        self.Y_test  = label[num_train:N,:]
        
        
        self.logger.info ("X_train shape: " + str(self.X_train.shape))
        self.logger.info ("Y_train shape: " + str(self.Y_train.shape))
        self.logger.info ("X_test shape: " + str(self.X_test.shape))
        self.logger.info ("Y_test shape: " + str(self.Y_test.shape))
        self.logger.info("total Images: "+str(self.X_train.shape[0]+self.X_test.shape[0])) 
        self.logger.info("dataset load done!")
        self.logger.info("\n")
        
        
    def visualize(self,log_dir):
        os.system('"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" http://localhost:6006')
        os.system('"D:\\Users\\Leon_PC\\Anaconda3\\envs\\tensorflow_gpu\\Scripts\\tensorboard.exe" --logdir='+log_dir)
    
    def restore(self,input_checkpoint_dir):
        self.logger.info("Restoring model starts...")
        checkpoints = pathlib.Path(input_checkpoint_dir).glob("*.ckpt")
        checkpoints = sorted(checkpoints, key=lambda cp:cp.stat().st_mtime)
        checkpoints = [cp.with_suffix('') for cp in checkpoints]
        latest = str(checkpoints[-1])+'.ckpt'
        self.logger.info("Restoring: "+latest)
        
        self.model.load_weights(latest)
        
        self.logger.info("Restoring model done.\n")
        file = h5py.File(input_checkpoint_dir+'/traindata.h5','r')
        self.X_train = file['trainx'][:]
        self.Y_train = file['trainy'][:] 
        file.close()
        
        file = h5py.File(input_checkpoint_dir+'/testdata.h5','r')
        self.X_test = file['testx'][:]
        self.Y_test = file['testy'][:] 
        file.close()

    def train(self): 
        self.model.fit(self.X_train, self.Y_train, 
                       epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose, 
                       validation_data=(self.X_test, self.Y_test),
                       callbacks=[self.history,self.tensorboard,self.cp_callback])
        self.history.loss_plot('epoch')
        self.test()

        
    def test(self):
        preds_train = self.model.evaluate(self.X_train, self.Y_train)
        self.logger.info("Train Loss = " + str(preds_train[0]))
        self.logger.info("Train Accuracy = " + str(preds_train[1]))
        
        preds_test  = self.model.evaluate(self.X_test, self.Y_test)
        self.logger.info("Test Loss = " + str(preds_test[0]))
        self.logger.info("Test Accuracy = " + str(preds_test[1]))
        
        
def mytrain():
    ImRec = ImRecognition('train')
    ImRec.train() 
#    ImRec.visualize(ImRec.log_dir)
    
def mytest(input_checkpoint_dir):
    ImRec = ImRecognition('test')
    ImRec.restore(input_checkpoint_dir)
    ImRec.test()
#    ImRec.visualize(input_checkpoint_dir)
     

if __name__ == '__main__':
    mytrain()
#    mytest("./logs/DB2-Conv2D-2020-07-20T16-26-55")
  
