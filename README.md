基于深度学习的图像识别框架
包含2个核心文件（分别是main.py、models文件夹、getNinaProData.py）

main.py包含核心框架和主函数。

models文件夹中包含网络模型。           
      
getNinaProData.py包含各种获取、整理、保存数据的函数。   
        
训练过程中的训练参数，网络模型、测试结果等，会保存在mian.log日志文件中，可以在每次训练/测试结束后查看。

该项目用于sEMG表面肌电信号识别，但是框架是通用的，只要修改一下getNinaProData.py，就可以用于其他识别应用。

本项目采用的数据是NinaPro DB2数据集，s1~s40，50个手势分类，3次实验

数据名称格式是Sx_Ey_A1.mat，其中x=1-40代表人的编号，y=1-3代表实验编号

数据集下载链接（需要注册账号）：http://ninapro.hevs.ch/

| index | train_acc | test_acc | network |
| ----- | ----- | ----- | ----- |
| 1      |  0.9803     |  0.9457    |  MobileNetv3_large|
| 2      |  0.9782     |  0.9344    |  MobileNetv1 |
| 3      |  0.9451     |  0.8284    |  ShuffleNetv2  |
   
