'''
Descripttion: 
version: 
Author: Yueyang
email: 1700695611@qq.com
Date: 2021-05-11 16:03:26
LastEditors: Yueyang
LastEditTime: 2021-05-11 17:55:47
'''
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
IMAGE_SIZE = 28

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


cv2.imshow('exp1',train_images[0])
cv2.imshow('exp2',train_images[1])
cv2.imshow('exp3',train_images[2])

model = tf.keras.models.load_model('model.h5')
model.summary()

predictions = model.predict(test_images)
print(predictions)
print('预测： '+class_names[np.argmax(predictions[0])])
print(predictions[0])
print(np.argmax(predictions[0]))
print(test_labels[0])
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(0, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(0, predictions,  test_labels)
plt.show()





test_img= cv2.imread('F:/SDK/TensorFlow/src/1.jpg')
gray= cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
img28= cv2.resize(gray,(28,28))
print(img28.shape)
plt.figure()
plt.imshow(img28)
plt.colorbar()
plt.grid(False)
plt.show()
img28=img28.reshape((1,28,28))
cv2.imshow('win',img28[0])
img28 = img28 / 255.0
print(img28.shape)
prediction = model.predict(img28)
print(prediction)
print('预测： '+class_names[np.argmax(prediction[0])])

cv2.waitKey(0)