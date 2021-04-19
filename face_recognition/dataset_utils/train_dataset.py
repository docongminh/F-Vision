import os
import random
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import logging as logger


def split_dataset(data_root): 
    """
    args: 
        data_root(str): path of root dataset
    returns: 
            train_list(list): list of path dataset
            num_class(int): number of class
    """
    train_list = []
    num_class = len(os.listdir(data_root))
    for class_name in os.listdir(data_root):
        if class_name.split('.')[-1] == 'txt':
            continue
        fps_class_name = os.path.join(data_root, class_name)
        for image in os.listdir(fps_class_name):
            image_name = os.path.join(class_name, image) 
            train_list.append((image_name, int(class_name)))

    print('number of class train dataset ', num_class)
    print('number of data point train dataset', len(train_list))
    return train_list, num_class


class ImageDataset(Dataset):
    """ 
    args:
            (object): 
            data_root(str): root path dataset. 
            image_shape(tupble): width*heigh of image. 
            crop_eye(bool): crop eye(upper face) as input or not.

    """ 
    def __init__(self, data_root, image_shape = (112,112), crop_eye=False):
        self.data_root = data_root
        self.train_list, self.num_class = split_dataset(data_root)
        self.crop_eye = crop_eye
        self.image_shape = image_shape
        
    def __len__(self):
        return len(self.train_list)
    def __num_class__(self): 
        return self.num_class

    def __getitem__(self, index):
        image_path, image_label = self.train_list[index]
        image_path = os.path.join(self.data_root, image_path)
        image = cv2.imread(image_path)

        if self.crop_eye:
            image = image[:60, :]
        
        image = cv2.resize(image, self.image_shape) #128 * 128
        
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
        
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        # normalize the image
        image = (image.transpose((2, 0, 1)) - 127.5) * 0.0078125
        image = torch.from_numpy(image.astype(np.float32))
        
        return image, image_label
   
class CommonTestDataset(Dataset):
    """ Data processor for model evaluation.

    Attributes:
        image_root(str): root directory of test set.
        image_list_file(str): path of the image list file.
        crop_eye(bool): crop eye(upper face) as input or not.
    """
    def __init__(self, image_root, image_list_file, crop_eye=False, image_shape = (112,112)):
        self.image_root = image_root
        self.image_list = []
        self.image_shape = image_shape 
        image_list_buf = open(image_list_file)
        line = image_list_buf.readline().strip()
        while line:
            self.image_list.append(line)
            line = image_list_buf.readline().strip()
        self.mean = 127.5
        self.std = 128.0
        self.crop_eye = crop_eye
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, index):
        short_image_path = self.image_list[index]
        image_path = os.path.join(self.image_root, short_image_path)
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        
        image = cv2.resize(image, self.image_shape)
        
        if self.crop_eye:
            image = image[:60, :]
        # normalize the image
        image = (image.transpose((2, 0, 1)) - self.mean) / self.std
        image = torch.from_numpy(image.astype(np.float32))
        
        return image, short_image_path


