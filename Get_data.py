#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/minkh93/MLDL/blob/master/Get_data.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


import cv2
import re
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder



class get_data:
    def __init__(self):
        print("access to data")
    
    def get_train_data():
        root_path = "/content/proj1/train/"
        
        train_input=[]
        train_label=[]

        for index in range(6):
            path = root_path+str(index)
            img_list = os.listdir(path)
            for img in img_list:
                image = cv2.imread(path+'/'+img, cv2.IMREAD_COLOR)
                train_input.append([np.array(image)])
                train_label.append([np.array(index)])
        return train_input, train_label
    
    def get_test_data():
        root_path = "/content/proj1/test"
        train_input=[]

        img_list = os.listdir(root_path)
        
        for img in img_list:
            image = cv2.imread("/content/proj1/test/"+img, cv2.IMREAD_COLOR)
            train_input.append([np.array(image)])
        return train_input

