# !/usr/bin/python
# -*- coding: utf-8 -*-

# @Time: 2020/2/9 下午1:39
# @Author: Casually
# @File: SVM_Train.py
# @Email: fjkl@vip.qq.com
# @Software: PyCharm

import cv2
import os
import numpy as np
from numpy.linalg import norm
from args import args

class StatModel(object):
    def load(self, fn):
        self.model = self.model.load(fn)

    def save(self, fn):
        self.model.save(fn)

class SVM(StatModel):
    def __init__(self, C=1, gamma=0.5):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)
        # 不能保证包括所有省份

    # 训练svm
    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    # 字符识别
    def predict(self, samples):
        r = self.model.predict(samples)
        return r[1].ravel()

# 定义参数
SZ = args.Size  # 训练图片长宽
MAX_WIDTH = args.MAX_WIDTH  # 原始图片最大宽度
Min_Area = args.Min_Area  # 车牌区域允许最大面积
PROVINCE_START = args.PROVINCE_START
provinces = args.provinces

# 来自opencv的sample，用于svm训练
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

# 来自opencv的sample，用于svm训练
def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n * ang / (2 * np.pi))
        bin_cells = bin[:10, :10], bin[10:, :10], bin[:10, 10:], bin[10:, 10:]
        mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps

        samples.append(hist)
    return np.float32(samples)

def train_svm(path):
    # 识别英文字母和数字
    Model = SVM(C=1, gamma=0.5)
    # 识别中文
    Modelchinese = SVM(C=1, gamma=0.5)
    # 英文字母和数字部分训练
    chars_train = []
    chars_label = []

    for root, dirs, files in os.walk(os.path.join(path,'chars')):
        if len(os.path.basename(root)) > 1:
            continue
        root_int = ord(os.path.basename(root))
        for filename in files:
            print('input：{}'.format(filename))
            filepath = os.path.join(root, filename)
            digit_img = cv2.imread(filepath)
            digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
            chars_train.append(digit_img)
            chars_label.append(root_int)

    chars_train = list(map(deskew, chars_train))
    chars_train = preprocess_hog(chars_train)
    chars_label = np.array(chars_label)
    Model.train(chars_train, chars_label)

    if not os.path.exists("svm.dat"):
        # 保存模型
        Model.save("svm.dat")
    else:
        # 更新模型
        os.remove("svm.dat")
        Model.save("svm.dat")

    # 中文部分训练
    chars_train = []
    chars_label = []

    for root, dirs, files in os.walk(os.path.join(path,'charsChinese')):
        if not os.path.basename(root).startswith("zh_"):
            continue
        pinyin = os.path.basename(root)
        index = provinces.index(pinyin) + PROVINCE_START + 1  # 1是拼音对应的汉字
        for filename in files:
            print('input：{}'.format(filename))
            filepath = os.path.join(root, filename)
            digit_img = cv2.imread(filepath)
            digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
            chars_train.append(digit_img)
            chars_label.append(index)
    chars_train = list(map(deskew, chars_train))
    chars_train = preprocess_hog(chars_train)
    chars_label = np.array(chars_label)
    Modelchinese.train(chars_train, chars_label)

    if not os.path.exists("svmchinese.dat"):
        # 保存模型
        Modelchinese.save("svmchinese.dat")
    else:
        # 更新模型
        os.remove("svmchinese.dat")
        Modelchinese.save("svmchinese.dat")


if __name__ == '__main__':
    path = 'train'
    train_svm(path)
    print('完成')
