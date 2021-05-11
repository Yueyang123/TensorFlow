'''
Descripttion: 
version: 
Author: Yueyang
email: 1700695611@qq.com
Date: 2021-05-11 16:16:59
LastEditors: Yueyang
LastEditTime: 2021-05-11 16:25:13
'''
import cv2
cap = cv2.VideoCapture(0)
while(1):
    # get a frame
    ret, frame = cap.read()
    # show a frame
    cv2.imshow("capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('./0.jpg',frame)
        break
