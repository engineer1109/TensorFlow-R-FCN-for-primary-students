# TensorFlow-R-FCN-for-primary-students
## 1.Introduction
R-FCN Tensorflow version for primary students

Mainly refer to https://github.com/auroua/tf_rfcn and https://github.com/endernewton/tf-faster-rcnn

Thanks for auroua and endernewton's help.

This code is for primary students to use.

What is different from theirs?

1. Support Python3 Train and Test

2. Support Chinese Language of detecting result

3. More easy to use and small changes

![](https://github.com/engineer1109/TensorFlow-R-FCN-for-primary-students/blob/master/sampleImg/en/004545.jpg)  
![](https://github.com/engineer1109/TensorFlow-R-FCN-for-primary-students/blob/master/sampleImg/zh/004545.jpg)  

## Requirements
System: Unix or Unix-Like Platrform  
(Have only tested on linux-ubuntu, other unix system should be supported in theory)  
Python3  
Tensorflow 1.8+  
OpenCV-Python  
CUDA  
CUDNN  

## Usage  
You need to download weights first. See the following Models Download.
Inference
```
    git clone https://github.com/engineer1109/TensorFlow-R-FCN-for-primary-students.git  
    cd lib && make
    cd ..
    sh run.sh 
```
Train
```
    sh train.sh
```
Tensorboard viewer
```
    sh tensorboard.sh
```
## Dataset of Pascal Voc
[Pascal Voc 2007+2012](https://github.com/engineer1109/TensorFlow-R-FCN-for-primary-students/tree/master/data/VOCdevkit2007/VOC2007)  
Unzip it to ./data/VOCdevkit2007/VOC2007
## Models Download
Main Model
Unzip it to ./output

  Download | Path 
  ------------- | ------------- 
  Baidu Yun | https://pan.baidu.com/s/1-zAgNRavUJGX5ggmSH-TEQ  Code: b4vt 
  Google | https://drive.google.com/open?id=1iWMqZAX3wh-XN0o5pIGClvIx8wMPWWFT


Pretrained Model
Unzip it to ./data/imagenet_weights

  Download | Path 
  ------------- | ------------- 
  Baidu Yun | https://pan.baidu.com/s/1oSZMlwOaIFOuqrMeSIY-PQ  Code: qvos 
  wget | http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz

For wget, you need to change filename to res101.ckpt
```
    wget -v http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
    tar -xzvf resnet_v1_101_2016_08_28.tar.gz
    mv resnet_v1_101.ckpt res101.ckpt
```
## This is the first complete version. V1.0
