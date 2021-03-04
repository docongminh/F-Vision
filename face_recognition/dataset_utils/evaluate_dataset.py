import os
import io
import cv2
import sys 
import bcolz
import torch
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


plt.switch_backend('agg')


def get_val_pair(path, name):
    carray = bcolz.carray(rootdir = os.path.join(path,name), mode='r')
    print('check path of data ',os.path.join(path,'{}_list.npy'.format(name)))
    issame = np.load(os.path.join(path,'{}_list.npy'.format(name)))
    
    return carray, issame
    
def get_val_data(data_path, val_type):
    dataset, label = get_val_pair(data_path, val_type)

    return dataset, label   
   
# 
# if __name__ == '__main__':
# 
#     data_path = '../face_recognition/data'
#     dataset , label = get_val_data(data_path, 'agedb_30')
#     print(dataset)
# 
#     print('labels \n \n ')
# 
#     print(label)
