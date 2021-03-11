""" 
@author: Jun Wang
@date: 20201012
@contact: jun21wangustc@gmail.com
"""

import os
import sys
import argparse
import yaml
from prettytable import PrettyTable
from torch.utils.data import DataLoader
from lfw.pairs_parser import PairsParserFactory
from lfw.lfw_evaluator import LFWEvaluator
from utils.model_loader import ModelLoader
from utils.extractor.feature_extractor import CommonExtractor
sys.path.append('../FaceX-Zoo')
from data_processor.test_dataset import CommonTestDataset
from backbone.backbone_def import BackboneFactory

def accu_key(elem):
    return elem[1]

# support for dataset "lfw, cplfw, calfw, agedb, rfw_African, rfw_Asian, rfw_Caucasian, rfw_Indian.")
#  model: resnet 50 ir, 152 ir se, mobilefacenet 
if __name__ == '__main__':
    test_set = 'CALFW'
    data_conf_file = '../FaceX-Zoo/test_protocol/data_conf.yaml'
    backbone_type = 'MobileFaceNet'
    backbone_conf_file = '../FaceX-Zoo/test_protocol/backbone_conf.yaml'
    
    batch_size = 16 
    model_path = '/home/minglee/Documents/ML/FaceX-Zoo/trash/mobilefacenet'
    
    # parse config.
    with open(data_conf_file) as f:
        data_conf = yaml.load(f)[test_set]
        pairs_file_path = data_conf['pairs_file_path']
        cropped_face_folder = data_conf['cropped_face_folder']
        image_list_file_path = data_conf['image_list_file_path']
    # define pairs_parser_factory
    pairs_parser_factory = PairsParserFactory(pairs_file_path, test_set)
    # define dataloader
    data_loader = DataLoader(CommonTestDataset(cropped_face_folder, image_list_file_path, False), 
                             batch_size=batch_size, num_workers=4, shuffle=False)
    #model def
    backbone_factory = BackboneFactory(backbone_type, backbone_conf_file)
    model_loader = ModelLoader(backbone_factory)
    feature_extractor = CommonExtractor('cuda:0')
    lfw_evaluator = LFWEvaluator(data_loader, pairs_parser_factory, feature_extractor)
    if os.path.isdir(model_path):
        accu_list = []
        model_name_list = os.listdir(model_path)
        for model_name in model_name_list:
            if model_name.endswith('.pt'):
                model_path = os.path.join(model_path, model_name)
                model = model_loader.load_model(model_path)
                mean, std = lfw_evaluator.test(model)
                accu_list.append((os.path.basename(model_path), mean, std))
        accu_list.sort(key = accu_key, reverse=True)
    else:
        model = model_loader.load_model(model_path)
        mean, std = lfw_evaluator.test(model)
        accu_list = [(os.path.basename(model_path), mean, std)]
    pretty_tabel = PrettyTable(["model_name", "mean accuracy", "standard error"])
    for accu_item in accu_list:
        pretty_tabel.add_row(accu_item)
    print(pretty_tabel)
