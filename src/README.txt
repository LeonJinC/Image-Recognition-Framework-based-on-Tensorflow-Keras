�������ѧϰ��ͼ��ʶ����
����2�������ļ����ֱ���main.py��models�ļ��С�getNinaProData.py��

main.py���Ŀ�ܺ�������
main.py---------class ImRecognition�����Ŀ�ܣ�------def __init__     ��ʼ�� �����������ʧ������ѵ���Ż����ȵ�
           |                                    |----def build_model  ����medel.py�е�ģ�ͣ�����tf�е�graph
           |                                    |----def visualize    ���ӻ�
           |                                    |----def restore      ģ������
           |                                    |----def train        ѵ������
           |                                    |----def test         ���Է���
           |                                    |----def getData      ��ȡ����
           |                        
           |----def mytrain() ����ʵ����class ImRecognition������ѵ��
           |----def mytest()  ����ʵ����class ImRecognition������ѵ���õ�ģ�ͣ�Ȼ����в���

models�ļ����а�������ģ��           
      
getNinaProData.py�������ֻ�ȡ�������������ݵĺ���         
getData.py-------def get_norm_data
           |-----def get_img_data 
           |-----def getnormdata    
           
ѵ�������е�ѵ������������ģ�͡����Խ���ȣ��ᱣ����mian.log��־�ļ��У�������ÿ��ѵ��/���Խ�����鿴��

����Ŀ����sEMG���漡���ź�ʶ�𣬵��ǿ����ͨ�õģ�ֻҪ�޸�һ��getNinaProData.py���Ϳ�����������ʶ��Ӧ��

����Ŀ���õ�������NinaPro DB2���ݼ���s1~s40��50�����Ʒ���

--------------------------------------------------------
 index  |  train_acc  |  test_acc  |  network
--------------------------------------------------------
 1      |  0.9803     |  0.9457    |  MobileNetv3_large 
 2      |  0.9782     |  0.9344    |  MobileNetv1
 3      |  0.9451     |  0.8284    |  ShuffleNetv2              
--------------------------------------------------------
   
