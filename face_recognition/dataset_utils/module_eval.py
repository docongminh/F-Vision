import torch 
import os 
import numpy as np 
from dataset_utils.train_dataset import CommonTestDataset 
from dataset_utils.extractor_embedding import CommonExtractor 
from dataset_utils.pairs_parser import PairsParserFactory
from dataset_utils.evaluator_dataset import Evaluator 
from torch.utils.data import DataLoader


import time
import warnings
import yaml 
from prettytable import PrettyTable

warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


class ModuleEval(): 
    """
    """ 
    def __init__(self, dataset_conf_yaml, model, batch_size, dataset_type): 
        self.model = model 
        if torch.cuda.is_available(): 
            model = torch.nn.DataParallel(self.model).cuda()
        print('data path', dataset_conf_yaml)
        self.dataset_conf_yaml = dataset_conf_yaml
        self.batch_size = batch_size
        self.dataset_type = dataset_type

    def eval(self): 
        inf_list = []
        feature_extractor = CommonExtractor(device,type='eval') 

        with open(self.dataset_conf_yaml) as f: 
            dataset_paths = yaml.load(f) 

        for data_type in self.dataset_type: 
            print('\nload eval dataset {} done !\n'.format(data_type))
            t = time.time()
            pairs_file =            dataset_paths[data_type]['pairs_file_path']
            cropped_foler_img =     dataset_paths[data_type]['cropped_face_folder']
            image_list_label_path = dataset_paths[data_type]['image_list_file_path'] 

            pairs_parser_factory = PairsParserFactory(pairs_file, data_type)
            data_loader = DataLoader(CommonTestDataset(cropped_foler_img, image_list_label_path, False),
                            batch_size=self.batch_size, num_workers = 4, shuffle=False)
            feature_extractor = CommonExtractor(device, type='eval')
            evaluator_dataset = Evaluator(data_loader, pairs_parser_factory, feature_extractor) 
            mean_acc, mean_tpr, mean_fpr ,std = evaluator_dataset.eval(self.model)

            inf_list.append((mean_acc, mean_tpr, mean_fpr, std, time.time()-t, data_type))
            

        pretty_tabel = PrettyTable(["mean accuracy","mean tpr","mean fpr" , "standard error", "time processing", "dataset type"])
        for row in inf_list:
            pretty_tabel.add_row(row)
        print(pretty_tabel)       

