import os
import random
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

def de_preprocess(tensor):

    return tensor * 0.5 + 0.5

hflip = transforms.Compose([
            de_preprocess,
            transforms.ToPILImage(),
            transforms.functional.hflip,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])


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
   

class NormBatch():
    """
    normalize batch size evaluate dataset 
    """
    def normalize_batch(self, tensor_batch): 
        norm_tensor = torch.empty_like(tensor_batch)
        print(tensor_batch.shape)
        for i, img in enumerate(tensor_batch):
            print('debug')
            print(img.shape)
            imm = img.permute(1,2,0)
            print(imm.shape)
            norm_img = normalize_image(img)
            norm_tensor[i] = norm_img
            
        return norm_tensor
    
    def l2_norm(self, input, axis = 1):
        norm = torch.norm(input, 2, axis, True)
        output = torch.div(input, norm)

        return output

    def hflip_batch(self, imgs_tensor):
        hfliped_imgs = torch.empty_like(imgs_tensor)
        for i, img_ten in enumerate(imgs_tensor):
            hfliped_imgs[i] = hflip(img_ten)

        return hfliped_imgs


