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
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataset_utils.train_dataset import EvaluateDataset

plt.switch_backend('agg')






sys.path.append('../face_recognition')
import config
    
def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)

def get_paths(lfw_dir, pairs):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
            issame = True
        elif len(pair) == 4:
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            path_list += (path0,path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs>0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)
    
    return path_list, issame_list 

def add_extension(path):
    if os.path.exists(path+'.jpg'):
        return path+'.jpg'
    elif os.path.exists(path+'.png'):
        return path+'.png'
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)

def get_paths_issame_ca_or_cp_lfw(lfw_dir, lfw_pairs):

    pairs = []
    with open(lfw_pairs, 'r') as f:
        for line in f.readlines()[0:]:
            pair = line.strip().split()
            pairs.append(pair)
    arr = np.array(pairs)

    paths = []
    actual_issame = []
    for count, person in enumerate(arr, 1): # Start counting from 1
        if count % 2 == 0:
            first_in_pair = arr[count-2]
            second_in_pair = person

            dir = os.path.expanduser(lfw_dir)
            path1 = os.path.join(dir, first_in_pair[0])
            path2 = os.path.join(dir, second_in_pair[0])
            paths.append(path1)
            paths.append(path2)

            if first_in_pair[1] != '0':
                actual_issame.append(True)
            else:
                actual_issame.append(False)
    
    return paths, actual_issame

def parse_dif_same_file(filepath):
    pairs_arr = []
    with open(filepath, 'r') as f:
        for line in f.readlines()[0:]:
            pair = line.strip().split(',')
            pairs_arr.append(pair)
    return pairs_arr 
# LFW
def get_paths_issame_LFW(lfw_dir):
    lfw_images_dir = lfw_dir + '/images'
    lfw_pairs = lfw_dir + '/pairs_LFW.txt'

    # Read the file containing the pairs used for testing
    pairs = read_pairs(os.path.expanduser(lfw_pairs))

    # Get the paths for the corresponding images
    paths, actual_issame = get_paths(os.path.expanduser(lfw_images_dir), pairs)

    return paths, actual_issame
# CPLFW
def get_paths_issame_CPLFW(cplfw_dir):
    cplfw_images_dir = cplfw_dir + '/images'
    cplfw_pairs = cplfw_dir + '/pairs_CPLFW.txt'
    return get_paths_issame_ca_or_cp_lfw(cplfw_images_dir, cplfw_pairs)  
# CALFW
def get_paths_issame_CALFW(calfw_dir):
    calfw_images_dir = calfw_dir + '/images'
    calfw_pairs = calfw_dir + '/pairs_CALFW.txt'
    return get_paths_issame_ca_or_cp_lfw(calfw_images_dir, calfw_pairs)
# CFP_FF and CFP_FP
def get_paths_issame_CFP(cfp_dir, type='FF'):

    pairs_list_F = cfp_dir + '/Pair_list_F.txt'
    pairs_list_P = cfp_dir + '/Pair_list_P.txt'

    path_hash_F = {}
    with open(pairs_list_F, 'r') as f:
        for line in f.readlines()[0:]:
            pair = line.strip().split()
            path_hash_F[pair[0]] = cfp_dir + '/' + pair[1]

    path_hash_P = {}
    with open(pairs_list_P, 'r') as f:
        for line in f.readlines()[0:]:
            pair = line.strip().split()
            path_hash_P[pair[0]] = cfp_dir + '/' + pair[1]


    paths = []
    actual_issame = []

    if type == 'FF':
        root_FF_or_FP = cfp_dir + '/Split/FF'
    else:
        root_FF_or_FP = cfp_dir + '/Split/FP'


    for subdir, _, files in os.walk(root_FF_or_FP):
        for file in files:
            filepath = os.path.join(subdir, file)

            pairs_arr = parse_dif_same_file(filepath)
            for pair in pairs_arr:
            
                first = path_hash_F[pair[0]]

                if type == 'FF':
                    second = path_hash_F[pair[1]]
                else:
                    second = path_hash_P[pair[1]]
                

                paths.append(first)
                paths.append(second)

                if file == 'diff.txt':
                    actual_issame.append(False)
                else:
                    actual_issame.append(True)

    return paths, actual_issame
   
def get_evaluate_dataset_and_loader(root_dir, type='LFW', num_workers=2, input_size=[112, 112], batch_size=100):
    
    if type == 'CALFW':
        paths, actual_issame = get_paths_issame_CALFW(root_dir)
    elif type == 'CPLFW':
        paths, actual_issame = get_paths_issame_CPLFW(root_dir)
    elif type == 'CFP_FF':
        paths, actual_issame = get_paths_issame_CFP(root_dir, type='FF')
    elif type == 'CFP_FP':
        paths, actual_issame = get_paths_issame_CFP(root_dir, type='FP')
    else:
        paths, actual_issame = get_paths_issame_LFW(root_dir)

    dataset = EvaluateDataset(data_root = paths, actual_issame=actual_issame, image_shape=input_size)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
    return dataset, data_loader
   
# if __name__ == '__main__':
# 
#     validation_paths_dic = {
#                 "LFW" : os.path.join(config.evaluate_dataset_root, 'lfw_112'),
#                 "CALFW" : os.path.join(config.evaluate_dataset_root, 'calfw_112'),
#                 "CPLFW" : os.path.join(config.evaluate_dataset_root, 'cplfw_112'),
#                 "CFP_FF" : os.path.join(config.evaluate_dataset_root, 'cfp_112'),
#                 "CFP_FP" : os.path.join(config.evaluate_dataset_root, 'cfp_112')}
#     validation_data_dic = {}
#     for val_type in config.validations:        
#         print('Init dataset and loader for validation type: {}'.format(val_type))
# 
#         dataset, loader = get_evaluate_dataset_and_loader(root_dir=validation_paths_dic[val_type], 
#                                                                 type=val_type, 
#                                                                 num_workers=config.num_workers, 
#                                                                 input_size=config.image_shape, 
#                                                                 batch_size=config.evaluate_batch_size)
#         print(dataset, loader)


