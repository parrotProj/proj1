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
    
    def train():
        root_path = "/content/proj1/train/"
        train_input=[]
        train_label=[]

        for index in range(6):
            path = root_path+str(index)
            img_list = os.listdir(path)
            for img in img_list:
                image = cv2.imread(path+'/'+img, cv2.IMREAD_COLOR)
                train_input.append(image)
                train_label.append(index)
        return np.array(train_input), train_label
    
    def test():
        root_path = "/content/proj1/test"
        train_input=[]
        img_name=[]

        img_list = os.listdir(root_path)
        
        for img in img_list:
            image = cv2.imread("/content/proj1/test/"+img, cv2.IMREAD_COLOR)
            train_input.append(image)
            img_name.append(img)
        return np.array(train_input),np.array(img_name)

'''
class image_pre:
    def randomCrop(img, width, height):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
      
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y+height, x:x+width]

        return img

    def make_input_batch(batchsize,imageArray):
'''     
