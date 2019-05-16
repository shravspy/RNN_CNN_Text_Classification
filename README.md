# RNN_CNN_Text_Classification: 

This Project was initiatied to implement TopicCNN, a convolutional neural network (CNN) based language model designed specifically for text classification via latent topics. 
Our motivation comes from TopicRNN, a recurrent neural network (RNN)-based language model. 

Below is the command to run: 

python train.py --data_file=./data/data.csv --clf=lstm

to Test you need to run :

python test.py --test_data_file=./data/data.csv --run_dir=./runs/1111111111 --checkpoint=clf-10000


Requirements: 

Python 3.x
Tensorflow > 1.5
Sklearn > 0.19.0
