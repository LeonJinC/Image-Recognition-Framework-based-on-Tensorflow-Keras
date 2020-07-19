

import numpy as np
import scipy.io as scio

'''
该模块用于读取所有手势的数据，且数据格式为肌电通道数*滑动窗口长度*1，存在数据量不足导致错误的bug。
'''

'''
1.函数说明：将需要转换成图像格式的数据进行归一化处理
2.函数参数：原始肌电数据、第几个手势数据、重复次数数据、开始通道1、结束通道2、手势总数、训练批次
3.返回值：归一化后的待处理图像数据
'''  
def get_norm_data(emg, stimulus, repetition, channel1, channel2, gesture1, gesture2, trainlist):
    # 训练批次
    train = len(trainlist)
    # 原始数据总集合
    emgset = np.zeros(((gesture2-gesture1+1)*train*10000, channel2-channel1+1))
    # 第几个手势
    for gesture_num in range(gesture1, gesture2+1):
        count = 0
        # 第几次训练
        for train_num in trainlist:
            # 第几个手势，第几次训练，对应通道的数据，并且取前10000个采样点
            all_ch_emg = emg[(stimulus[:,0] == gesture_num) & (repetition[:,0] == train_num), channel1-1:channel2][0:10000]
            # 均值、标准差归一化
            for ch in range(emgset.shape[1]):
                ch_data = all_ch_emg[:,ch]
                avg = np.average(ch_data)
                std = np.std(ch_data)
                norm_ch_data = (ch_data-avg)/std
                all_ch_emg[:,ch] = norm_ch_data
            # 数据填充
            emgset[(gesture_num - gesture1)*train*10000+count*10000:(gesture_num - gesture1)*train*10000+(count+1)*10000] = all_ch_emg 
            count += 1
    
    return emgset



'''
1.函数说明：将处理后的数据封装成图像格式
2.函数参数：处理后的数据、开始通道1、结束通道2、手势总数、训练批次、窗口长度、滑动步长
3.返回值：图像格式的数据和标签
'''
def get_img_data(data, channel1, channel2, gesture1, gesture2, trainlist, winlen, slidlen):
    # 训练总次数
    train = len(trainlist)
    # 一个手势训练一次产生的图片总数
    img_num = int((10000-winlen)/slidlen)+1
    # 单个图片数据
    img = np.zeros((channel2-channel1+1, winlen, 1)) 
    # 所有图片数据
    all_imgs = np.zeros(((gesture2-gesture1+1)*train*img_num, channel2-channel1+1, winlen, 1))
    # 标签
    labels = np.zeros(((gesture2-gesture1+1)*train*img_num,))
    
    # 第几个手势
    for gesture_num in range(gesture1, gesture2+1):
        gesture_num = gesture_num - gesture1
        # 第几次训练
        for train_num in range(0, train):
            # 第几个手势，第几次训练的对应通道的数据
            all_ch_data = data[gesture_num*train*10000+train_num*10000: gesture_num*train*10000+(train_num+1)*10000]
            # 第几个图片数据
            for num in range(0, img_num):   
                # 图片填充
                sin_img_data = all_ch_data[num*slidlen: num*slidlen + winlen]
                img[:,:,0] = sin_img_data.T
                # 数据归总                     
                all_imgs[gesture_num*train*img_num + train_num*img_num + num,:,:,:] = img
                labels[gesture_num*train*img_num + train_num*img_num + num] = int(gesture_num + gesture1 - 1)
                
    return all_imgs, labels


    
'''
1.函数说明：获取指定人数的图像数据和标签
2.函数参数：开始通道1、结束通道2、手势总数、训练批次、窗口长度、滑动步长、人员列表
3.返回值：图像格式的数据和标签
'''    
from tqdm import tqdm
def getnormdata(channel1, channel2, trainlist, winlen, slidlen, people):  
    x = np.zeros((1,1))
    y = np.zeros((1,1))
    for i in tqdm(people):
        for j in range(1,4):
            data = scio.loadmat('../DB2/S%d_E%d_A1.mat' %(i, j))
            emg = data['emg']
            repetition = data['repetition']
            stimulus = data['stimulus']        
            gesture1 = min(stimulus[:,0])
            gesture2 = max(stimulus[:,0])
            if j == 1:
                gesture1 = 1
            elif j == 2:
                gesture1 = 18
            else:
                gesture1 = 41
            onedata = get_norm_data(emg, stimulus, repetition, channel1, channel2, gesture1, gesture2, trainlist)
            temp_x, temp_y = get_img_data(onedata, channel1, channel2, gesture1, gesture2, trainlist, winlen, slidlen)
            if x.size == 1:
                x = temp_x
                y = temp_y
            else:
                x = np.concatenate((x,temp_x), axis = 0)
                y = np.concatenate((y,temp_y), axis = 0)
    
    return x, y


if __name__=='__main__':
    start=1     #从第1个人
    end=1      #到第30个人
    people=[i for i in range(start,end+1)]
#    print(people)
    trainx,trainy=getnormdata(1, 12, [1,3,4,6], 200, 200, people)
    
    print('trainx.shape: ',trainx.shape,'\n')
    print('trainy.shape: ',trainy.shape,'\n')
    
    import h5py
    print('saving ','../DB2/DB2_S'+str(start)+'-'+str(end)+'_norm_winlen200_slide200.h5')
    file = h5py.File('../DB2/DB2_S'+str(start)+'-'+str(end)+'_norm_winlen200_slide200.h5','w')  
    file.create_dataset('trainx', data = trainx)  
    file.create_dataset('trainy', data = trainy)  
    file.close() 
    
    
    
    
    
    
    

