基于深度学习的图像识别框架
包含2个核心文件（分别是main.py、models文件夹、getNinaProData.py）

main.py核心框架和主函数
main.py---------class ImRecognition（核心框架）------def __init__     初始化 输入输出、损失函数、训练优化器等等
           |                                    |----def build_model  调用medel.py中的模型，构建tf中的graph
           |                                    |----def visualize    可视化
           |                                    |----def restore      模型重载
           |                                    |----def train        训练方法
           |                                    |----def test         测试方法
           |                        
           |----def mytrain() 用来实例化class ImRecognition，进行训练
           |----def mytest()  用来实例化class ImRecognition，重载训练好的模型，然后进行测试

models文件夹中包含网络模型           
      
getNinaProData.py包含各种获取、整理、保存数据的函数         
getData.py-------def get_norm_data
           |-----def get_img_data 
           |-----def getnormdata    
           
训练过程中的训练参数，网络模型、测试结果等，会保存在mian.log日志文件中，可以在每次训练/测试结束后查看。

该项目用于sEMG表面肌电信号识别，但是框架是通用的，只要修改一下getNinaProData.py，就可以用于其他识别应用

本项目采用的数据是NinaPro DB2数据集，s1~s40，50个手势分类

--------------------------------------------------------
 index  |  train_acc  |  test_acc  |  network
--------------------------------------------------------
 1      |  0.9803     |  0.9457    |  MobileNetv3_large 
 2      |  0.9782     |  0.9344    |  MobileNetv1
 3      |  0.9451     |  0.8284    |  ShuffleNetv2              
--------------------------------------------------------
