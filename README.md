Dec 26 2023

Qingjian Shi
qshi10@u.rochester.edu



_________________________________________________________________________________________________________________

**Citations for Resources Used:**

Nicholas Renotte
Deep Drowsiness Detection using YOLO, Pytorch and Python

TzuTa Lin
labelImg

Glenn Jocher
YOLOv5 by Ultralytics

_________________________________________________________________________________________________________________



**Install Dependencies:**

*****************************************************************************************************************

pip3 install torch torchvision 
!git clone https://github.com/ultralytics/yolov5.git 
!git clone https://github.com/HumanSignal/labelImg.git 
pip install -r requirements.txt 
pip install pyqt5 lxml --upgrade 
cd labelImg && pyrcc5 -o libs/resources.py resources.qrc 

*****************************************************************************************************************



**INCLUDE THIS AT TOP OF ALL CODE:**

*****************************************************************************************************************
import torch, matplotlib.pyplot as plt  
import numpy as np  

import cv2 , uuid,os ,time, ssl , certifi
from urllib import request

ssl._create_default_https_context = ssl._create_unverified_context

*****************************************************************************************************************



**Load Model:**
*****************************************************************************************************************

model =  torch.hub.load('ultralytics/yolov5', 'yolov5')
*****************************************************************************************************************



**To train custom model:**
*****************************************************************************************************************

First you have to upload pictures and store them in the images folder. Then you have to use labelImg to classify them as YOLO format, using terminal

python3 labelImg.py

set current folder as data/images dir, set saving folder as data/labels

crop area of each picture with intended area being valuable to train. Then train on dataset using:

python3 train.py --img 320 --batch 16 --epochs 500 --data dataset.yaml --weights yolov5n.pt --workers 0

This took .91 hours of runtime, completed after around 400 epochs. 
*****************************************************************************************************************


**To run:**
*****************************************************************************************************************
  python3 tutorial.py

*****************************************************************************************************************

This is a live-feed deep detection program in python that analyzes trained on custom model using labeImg and YOLOv5




