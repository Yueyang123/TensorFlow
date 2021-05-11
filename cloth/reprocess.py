'''
Descripttion: 
version: 
Author: Yueyang
email: 1700695611@qq.com
Date: 2021-05-10 20:44:29
LastEditors: Yueyang
LastEditTime: 2021-05-11 16:43:00
'''
import cv2
 
test_img= cv2.imread('F:/SDK/TensorFlow/src/1.jpg')
gray= cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
cv2.imshow('win1',gray)
img28= cv2.resize(gray,(28,28))
cv2.imshow('win2',img28)
cv2.waitKey(0)

