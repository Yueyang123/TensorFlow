'''
Descripttion: 
version: 
Author: Yueyang
email: 1700695611@qq.com
Date: 2021-05-11 21:22:37
LastEditors: Yueyang
LastEditTime: 2021-05-11 21:40:05
'''
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import matplotlib.pyplot as plt
font = cv2.FONT_HERSHEY_SIMPLEX
model =keras.models.load_model('num_model.h5') #读取网络
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
cap = cv2.VideoCapture(1)
while(1):
    # get a frame
    ret, image = cap.read()
    image_ =  cv2.resize(image, (250,250), interpolation = cv2.INTER_AREA)
    image = cv2.cvtColor(image_, cv2.COLOR_BGR2GRAY)#灰度化处理
    img_w = cv2.Sobel(image,cv2.CV_16S,0,1)#Sobel滤波，边缘检测
    img_h = cv2.Sobel(image,cv2.CV_16S,1,0)#Sobel滤波，边缘检测
    img_w = cv2.convertScaleAbs(img_w)
    _, img_w = cv2.threshold(img_w,0,255,cv2.THRESH_OTSU|cv2.THRESH_BINARY)
    img_h = cv2.convertScaleAbs(img_h)
    _, img_h = cv2.threshold(img_h,0,255,cv2.THRESH_OTSU|cv2.THRESH_BINARY)
    image = img_w + img_h
    image = cv2.morphologyEx(image,cv2.MORPH_CLOSE,kernel)
    temp_data = np.zeros((250,10))
    image = np.concatenate((temp_data,image,temp_data),axis = 1)
    temp_data = np.zeros((10,270))
    image = np.concatenate((temp_data,image,temp_data),axis = 0)
    image = cv2.convertScaleAbs(image)
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for _ in contours:
        x,y,w,h = cv2.boundingRect(_)
        if w*h  < 100:
            continue
        img_model = image[y-10:y+h+10,x-10:x+w+10]
        img_model =  cv2.resize(img_model, (28,28), interpolation = cv2.INTER_AREA)
        img_model = img_model/255
        predict = model.predict(img_model.reshape(-1,28,28,1))
        if np.max(predict) > 0.5:
            data_predict = str(np.argmax(predict))
            image_z = cv2.rectangle(image_,(x-10,y-10),(x + w-10,y + h-10),(255,0,0),1)
            image_z = cv2.putText(image_z,data_predict , (x+10, y+10), font, 0.7, (0, 0, 255), 1)
    cv2.imshow("capture", image_z)
    cv2.waitKey(30)


