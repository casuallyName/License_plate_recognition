# coding=gbk

import cv2
import numpy as np

img = cv2.imread('1.jpg')
cv2.imshow('card',img)
img = cv2.imread('1.jpg',0)
cv2.imshow('gary',img)
height,width = img.shape
thres,binary = cv2.threshold(img,0,255,cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
cv2.imshow('threshold',binary)
# print(img.shape)
paint = np.zeros(img.shape,dtype=np.uint8)
# 每一列黑色像素个数
pointSum = np.zeros(width,dtype=np.uint8)
for x in range(width):
    for y in range(height):
        if binary[y][x]:
            pointSum[x] = pointSum[x] + 1


for x in range(width):
    for y in range(height)[::-1]:
        if (pointSum[x]):
            paint[y][x] = 255
            pointSum[x] = pointSum[x] - 1



cv2.imshow('paint',paint)
cv2.waitKey(0)