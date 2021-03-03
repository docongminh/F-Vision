import os
import random
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

def normalize_image(image): 
    image = (image.transpose((2, 0, 1)) - 127.5) * 0.0078125
    image = torch.from_numpy(image.astype(np.float32))
    return image

def split_dataset(data_root): 
    train_list = []
    num_class = len(os.listdir(data_root))
    for class_name in os.listdir(data_root):
        fps_class_name = os.path.join(data_root, class_name)
        for image in os.listdir(fps_class_name):
            image_name = os.path.join(class_name, image) 
            train_list.append((image_name, int(class_name)))
    print('num_class ', num_class)
    print('num data point', len(train_list))
    return train_list, num_class


class ImageDataset(Dataset):
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
        norm_img = normalize_image(image)
        # inital zeros matrix for test
        # norm_img = torch.Tensor(np.zeros((3,112,112)))
        return norm_img, image_label
   

class EvaluateDataset(Dataset):
    def __init__(self, data_root, actual_issame, image_shape = (112,112), crop_eye=False):
        self.data_root = data_root
        self.actual_issame = actual_issame
        self.crop_eye = crop_eye
        self.image_shape = image_shape 

        self.nrof_embeddings = len(self.actual_issame)*2  # nrof_pairs * nrof_images_per_pair
        self.labels_array = np.arange(0,self.nrof_embeddings)
    def __len__(self):
        return len(self.data_root) 
        
    def __getitem__(self, index):
        image_path = self.data_root[index]
        image = cv2.imread(image_path)
        image_label = self.labels_array[index]
        
        if self.crop_eye:
            image = image[:60, :]
        image = cv2.resize(image, self.image_shape) #128 * 128
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        norm_img = normalize_image(image)
        return norm_img, image_label


