
import cv2
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder


class get_data:
    def __init__(self):

    def get_train_data():
        root_dir='/content/MLDL/train/'
        train_input=[]
        train_label=[]

        for index in range(6):
            path = root_dir+str(index)
            print(path)
            img_list = os.listdir(path)
            get_ipython().system('cd $path')
            for img in img_list:
                image = cv2.imread(str(index)+'/'+img, cv2.IMREAD_COLOR)
                train_input.append([np.array(image)])
                train_label.append([np.array(index)])
        return train_input, train_label
    
    def get_test_data():
        root_dir='/content/MLDL/train/'
        train_input=[]
        train_label=[]

    
        img_list = os.listdir(path)
        
        for img in img_list:
            image = cv2.imread(str(index)+'/'+img, cv2.IMREAD_COLOR)
            train_input.append([np.array(image)])
            train_label.append([np.array(index)])
        return train_input, train_label

