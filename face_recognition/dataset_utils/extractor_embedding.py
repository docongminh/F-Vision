"""
@author: Jun Wang
@date: 20201016 
@contact: jun21wangustc@gmail.com 
"""

import os
import logging as logger
import numpy as np
import torch
from tqdm import tqdm 
logger.basicConfig(level=logger.INFO, 
                   format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')

class CommonExtractor:
    """Common feature extractor.
    
    Attributes:
        device(object): device to init model.
    """
    def __init__(self, device, status = 'train'):
        self.device = torch.device(device)
        self.status = status
    def extract_online(self, model, data_loader):
        """Extract and return features.
        
        Args:
            model(object): initialized model.
            data_loader(object): load data to be extracted.
            status (str): set type of train or eval 
        Returns:
            image_name2feature(dict): key is the name of image, value is feature of image.
        """
        model.eval()
        image_name2feature = {}
        with torch.no_grad(): 
            for (images, filenames) in tqdm(data_loader):
                images = images.to(self.device)
                if self.status =='eval':
                    features = model.forward(images).cpu().numpy()
                else:               
                    features = model.forward(images, None,status='eval').cpu().numpy()
                    
                for filename, feature in zip(filenames, features): 
                    image_name2feature[filename] = feature
        return image_name2feature


class FeatureHandler:
    """Some method to deal with features.
    
    Atributes:
        feats_root(str): the directory which the fetures in.
    """
    def __init__(self, feats_root):
        self.feats_root = feats_root

    def load_feature(self):
        """Load features to memory.
        
        Returns:
            image_name2feature(dict): key is the name of image, value is feature of image.
        """
        image_name2feature = {}
        for root, dirs, files in os.walk(self.feats_root):
            for cur_file in files: 
                if cur_file.endswith('.npy'):
                    cur_file_path = os.path.join(root, cur_file)
                    cur_feats = np.load(cur_file_path)
                    if self.feats_root.endswith('/'):
                        cur_short_path = cur_file_path[len(self.feats_root) : ]
                    else:
                        cur_short_path = cur_file_path[len(self.feats_root) + 1 : ]
                    cur_key = cur_short_path.replace('.npy', '.jpg')
                    image_name2feature[cur_key] = cur_feats
        return image_name2feature
