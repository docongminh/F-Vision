
import os
import sys
import argparse
from torch.utils.data import DataLoader
from utils.model_loader import ModelLoader
from backbone.backbone_def import BackboneFactory
from dataset_utils.module_eval import ModuleEval
from prettytable import PrettyTable
import config as conf

if __name__ == '__main__':
    backbone_parametor = conf.model_parameter[conf.backbone_type]  
    backbone_factory = BackboneFactory(conf.backbone_type, backbone_parametor) 
    model_loader = ModelLoader(backbone_factory, conf.device) 
    model = model_loader.load_model(conf.pretrain_model)
    f = open(conf.result_test_file, "a") 
    f.write('\n\n')
    f.write('\nbackbone: {}: model_parametor:{}\n'.format(conf.backbone_type, backbone_parametor))
    
    if conf.root_eval_dataset.split('/')[-1] == "VN_celeb": 
        f.write("dataset without mask \n")
    else:
        f.write('dataset with mask \n')
    
    print('load model done !')
    print('backbone: {}: model_parametor:{}'.format(conf.backbone_type, backbone_parametor))
    evaluate_dataset = ModuleEval(conf.root_eval_dataset, model, conf)
    
    evaluate_dataset.eval()
    f.close
