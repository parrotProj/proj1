#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/minkh93/MLDL/blob/master/Get_data.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


import cv2
import re
import numpy as np
import pandas as pd
import os
import random

from skimage.transform import resize
from sklearn.preprocessing import OneHotEncoder

def train(size,divide=False):
    print("input_size : ", size)
    root_path = "/content/proj1/train/"
    
    train_input=[]
    train_label=[]

    for index in range(6):
        path = root_path+str(index)
        img_list = os.listdir(path)
        for img in img_list:
            image = cv2.imread(path+'/'+img, cv2.IMREAD_COLOR)
            
            train_input.append(cv2.resize(image,(size,size)))
            train_label.append(index)
    if divide==True:
        return np.divide(np.array(train_input),255.), np.array(train_label)
    return np.array(train_input), np.array(train_label)

def test(size,divide=False):
    root_path = "/content/proj1/test"
    test_input=[]

    img_list = os.listdir(root_path)

    for img in img_list:
        image = cv2.imread("/content/proj1/test/"+img, cv2.IMREAD_COLOR)
        test_input.append(cv2.resize(image,(size,size)))
    name = pd.DataFrame(img_list)
    name = name[0].apply(lambda x:x.split('.')[0])
    if divide == True:
        return np.divide(np.array(test_input),255.), np.array(name)
    return np.array(test_input), np.array(name)
