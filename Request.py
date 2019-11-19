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



class get_data:
    def __init__(self, size=150):
        self.size = 150;
        print("access to data")
    
    def train():
        root_path = "/content/proj1/train/"
        
        train_input=[]
        train_label=[]

        for index in range(6):
            path = root_path+str(index)
            img_list = os.listdir(path)
            for img in img_list:
                image = cv2.imread(path+'/'+img, cv2.IMREAD_COLOR)
                
                train_input.append(resize(image,(size,size,3)))
                train_label.append(index)
        return np.array(train_input), train_label
    
    def test():
        root_path = "/content/proj1/test"
        test_input=[]

        img_list = os.listdir(root_path)
        
        for img in img_list:
            image = cv2.imread("/content/proj1/test/"+img, cv2.IMREAD_COLOR)
            test_input.append(reszie(image,(size,size,3)))
        return np.array(test_input), img_list


