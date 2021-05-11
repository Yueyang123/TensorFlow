'''
Descripttion: 
version: 
Author: Yueyang
email: 1700695611@qq.com
Date: 2021-05-11 20:23:04
LastEditors: Yueyang
LastEditTime: 2021-05-11 21:12:20
'''
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class_names = ['CAT','DOG']
model = tf.keras.models.load_model('F:/SDK/TensorFlow/dogcat/train/model.h5')
model.summary()

def _pre_read(image_filename):
    #读取图片
    image=tf.io.read_file(image_filename)
    #解码图片
    image=tf.image.decode_jpeg(image,channels=3)
    #转换图片的大小
    image=tf.image.resize(image,(200,200))
    image=tf.reshape(image,[200,200,3])
    #图片归一化
    image=image//255
    return image
image= _pre_read('F:/SDK/TensorFlow/dogcat/img/dog.12432.jpg')
X = np.zeros((1, 200, 200, 3), dtype='float32')
X[0] = image
prediction = model.predict(X)
print(prediction[0])
if(prediction[0][0]<0.5):
    print('预测： '+class_names[1])
else:
    print('预测： '+class_names[0])
