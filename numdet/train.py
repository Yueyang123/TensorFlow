'''
Descripttion: 
version: 
Author: Yueyang
email: 1700695611@qq.com
Date: 2021-05-11 21:22:12
LastEditors: Yueyang
LastEditTime: 2021-05-11 21:22:27
'''
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from tensorflow import keras
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
def show_single_image(img_arr):
    img_arr = img_arr.reshape(28,28)
    plt.imshow(img_arr,cmap = "binary")
    plt.show()
x = 0
show_single_image(x_train[x])
x_train = x_train/255.
x_test = x_test/255.
model = keras.models.Sequential() #先生成一个模型框架
model.add(keras.layers.Conv2D(filters = 128,
                              kernel_size = 3,
                              padding = 'same',
                              activation = 'selu',
                              input_shape = (28,28,1)
                              ))
model.add(keras.layers.SeparableConv2D(filters = 128,
                              kernel_size = 3,
                              padding = 'same',
                              activation = 'selu',
                              ))
model.add(keras.layers.MaxPool2D(pool_size = 2))
model.add(keras.layers.SeparableConv2D(filters = 256,
                              kernel_size = 3,
                              padding = 'same',
                              activation = 'selu',
                              ))
model.add(keras.layers.SeparableConv2D(filters = 256,
                              kernel_size = 3,
                              padding = 'same',
                              activation = 'selu',
                              ))
model.add(keras.layers.MaxPool2D(pool_size = 2))
model.add(keras.layers.SeparableConv2D(filters = 512,
                              kernel_size = 3,
                              padding = 'same',
                              activation = 'selu',
                              ))
model.add(keras.layers.SeparableConv2D(filters = 512,
                              kernel_size = 3,
                              padding = 'same',
                              activation = 'selu',
                              ))
model.add(keras.layers.MaxPool2D(pool_size = 2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128,activation = 'selu'))
model.add(keras.layers.Dense(10,activation = 'softmax'))
model.compile(optimizer='adam',#求解模型的方法
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
history = model.fit(x_train, y_train,epochs = 5, #history用来接收训练过程中的一些参数数值 #训练的参数#训练5遍
                    validation_data = (x_test,y_test) )#实时展示模型训练情况   
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1.5)
    plt.show()
plot_learning_curves(history)#（输入训练时的返回值）
def predict_data(test_data):
    pred = model.predict(test_data.reshape(-1,28,28,1))
    return np.argmax(pred)
show_single_image(x_test[0])
print("模型的预测结果是：",predict_data(x_test[0]))
model.save('num_model.h5')                             
