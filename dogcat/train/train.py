'''
Descripttion: 
version: 
Author: Yueyang
email: 1700695611@qq.com
Date: 2021-05-11 18:50:04
LastEditors: Yueyang
LastEditTime: 2021-05-11 19:11:07
'''
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import glob
import matplotlib.pyplot as plt

#使用tf.data来读取数据集 
#使用tf.keras来搭建网络

image_filenames=glob.glob("F:/SDK/TensorFlow/dogcat/img/*.jpg")  #读取train的所有图片，获取的图片的路径
#对路径进行乱序
image_filenames=np.random.permutation(image_filenames)

#此处lambda与map合用相当于：lambda函数用于指定对列表image_filenames中每一个元素的共同操作若==成立，表示当前标签为cat，label=1； 若当前标签为dog，则label=0。
train_labels = list(map(lambda x: float(x.split('\\')[1].split('.')[0] == 'cat'), image_filenames))
#这里的x其实就是后面的image_filenames(参考map函数和lambda函数的用法)

train_dataset=tf.data.Dataset.from_tensor_slices((image_filenames,train_labels)) #创建dataset

def _pre_read(image_filename,label):
    #读取图片
    image=tf.io.read_file(image_filename)
    #解码图片
    image=tf.image.decode_jpeg(image,channels=3)
    #转换图片的大小
    image=tf.image.resize(image,(200,200))
    image=tf.reshape(image,[200,200,3])
    #图片归一化
    image=image//255
    return image,label


train_dataset=train_dataset.map(_pre_read)   #对数据集进行图片预处理
train_dataset=train_dataset.shuffle(300)   #乱序
train_dataset=train_dataset.repeat()
train_dataset=train_dataset.batch(32)    
#建立好网络之后，直接从dataset中取读取 
#<BatchDataset shapes: ((None, 200, 200, 3), (None,)), types: (tf.float32, tf.float32)>
#创建模型
model=keras.Sequential()
model.add(layers.Conv2D(64,(3,3),activation="relu",input_shape=(200,200,3)))
model.add(layers.MaxPool2D(2,2))

model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Dropout(0.5))

model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(32,activation="relu"))
model.add(layers.Dense(1,activation="sigmoid"))

model.summary()

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["acc"])

#经过多少个step完成一个epoch，因为之前的数据集repeat()为无限次
train_step_per_epoch=len(image_filenames)//32
history=model.fit(train_dataset,epochs=30,steps_per_epoch=train_step_per_epoch)
history.history.keys()
plt.plot(history.epoch,history.history.get("loss"))
plt.plot(history.epoch,history.history.get("acc"))
print('SAVE MODEL!!!')
model.save('F:/SDK/TensorFlow/dogcat/train/model.h5')
print('SAVE MODEL SUCCESS!!!')

