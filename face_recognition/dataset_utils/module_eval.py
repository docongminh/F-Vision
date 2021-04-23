import torch 
import os 
import numpy as np 
import sys 
sys.path.append('../F-Vision')
from dataset_utils.train_dataset import CommonTestDataset 
from dataset_utils.extractor_embedding import CommonExtractor 
from dataset_utils.pairs_parser import PairsParserFactory
from dataset_utils.evaluator_dataset import Evaluator 
from torch.utils.data import DataLoader
from pathlib import Path 
import time
import warnings
from prettytable import PrettyTable
from backbone.backbone_def import BackboneFactory 
from utils.model_loader import ModelLoader 

warnings.filterwarnings("ignore", category=UserWarning)


class GenPairImage():
    """Common feature extractor.
        generate pair of image.
    """
    def __init__(self, root_dir, text_file_path, text_label_path, num_of_pair ):
        """Common feature extractor.
            args:
                root_dir(str): the root data path (must be contain folder with sub-folder)
                text_file_path(str): the path of text's file create labels of true's pairs and false's pairs
                text_label_path(str): the path of text's file create labels 
                num_of_pair(int): number of pair need to be create 
            returns: 
                    None. 
        """
        self.nofpair = int(num_of_pair) 
        self.root_dir = root_dir 
        self.text_file_path = text_file_path
        self.text_label_path = text_label_path
        # if not os.path.exists(text_file_path): 
        print('generate pair pairse file \n')
        self.num_of_pair_false()
        self.num_of_pair_true()
        # if not os.path.exists(text_label_path):
        print('generate label file \n')
        self.gen_label_file()

    def gen_label_file(self): 
        f = open(self.text_label_path,'w')
        for sub_folder in os.listdir(self.root_dir):
            if sub_folder.split('.')[-1] == 'txt': 
                continue 
            sub_list = os.listdir(os.path.join(self.root_dir,sub_folder))
            for img in sub_list: 
                line = str(sub_folder + '/' + img + '\n')
                f.write(line)
        f.close()
        
    def num_of_pair_false(self): 
    
        f = open(self.text_file_path, "w")

        folder_img = os.listdir(self.root_dir)
        sub_first = ""
        maxx = 0
        for item in folder_img: 
            if item.split('.')[-1] !='txt': 
                if len(os.listdir(self.root_dir + '/' + item)) > maxx: 
                    maxx = len(os.listdir(self.root_dir + '/' + item)) 
                    sub_first = item 
                
        cnt = 0
        for item in folder_img: 
            if item != sub_first and item.split('.')[-1] !='txt': 
                folder_first  = os.listdir(self.root_dir + '/' + sub_first)
                folder_second = os.listdir(self.root_dir + '/' + item)
                for img_first in folder_first: 
                    for img_second in folder_second: 
                        f.write(str( sub_first + '/' + img_first) + " " + str( item + '/' + img_second) + " " + str(0) + "\n")
                        cnt += 1
                        if cnt >= int(self.nofpair//2): 
                            return None

    def num_of_pair_true(self): 

        f = open(self.text_file_path, "a")
        
        folder_img = os.listdir(self.root_dir) 
        cnt = 0 
        for sub_folder in folder_img: 
            if sub_folder.split('.')[-1] == 'txt':  
                continue
            sub_list = os.listdir(self.root_dir + '/' + sub_folder)
            if len(sub_list) < 3: 
                continue
            for idxf, img_first in enumerate(sub_list): 
                for idxs, img_second in enumerate(sub_list): 
                    if idxs > idxf: 
                        f.write(str(sub_folder +'/'+img_first) + " " + str(sub_folder + '/' + img_second) + " " + str(1) + "\n")
                        cnt +=1 
                        if cnt >= int(self.nofpair//2): 
                            return None



class ModuleEval():  
    """ComModuleEval feature extractor.
    
    Attributes:
        device(object): device to init model.
    """
    def __init__(self, data_path, model, conf, gen_pair=True, status='eval'): 
        """
        Args: 
            data_path(str): the paths contain data root 
            model (torch.nn): the backbone with weight of models. 
            conf (object): the object of defind configure 
            gen_pair (bool): status generate pair image or not 
            status (str): status of evaluate model (support for eval or train)
        returns: 
                None
        """
        self.model = model 
        if torch.cuda.is_available(): 
            model = torch.nn.DataParallel(self.model).cuda()
        self.conf = conf
        self.data_path = data_path
        self.text_file_path = self.data_path + '/' + 'pairs_file_path.txt'
        self.text_label_path = self.data_path + '/' + 'image_list_file_path.txt'
        self.distance_metric = conf.distance_metric
        self.result_test_file = conf.result_test_file
        self.status = status
        if gen_pair:
            GenPairImage(self.data_path, self.text_file_path, self.text_label_path, conf.num_of_pair)


    def eval(self): 
        f = open(self.result_test_file, "a") 

        feature_extractor = CommonExtractor(self.conf.device) 
        print("load eval dataset ! \n")    
        t = time.time()
        pairs_file =            self.text_file_path
        cropped_foler_img =     self.data_path
        image_list_label_path = self.text_label_path 
        pairs_parser_factory = PairsParserFactory(pairs_file)

        data_loader = DataLoader(CommonTestDataset(cropped_foler_img, image_list_label_path, False),batch_size= self.conf.evaluate_batch_size, num_workers = self.conf.num_workers, shuffle=False)
        
        feature_extractor = CommonExtractor(self.conf.device, status = self.status)
        
        evaluator_dataset = Evaluator(data_loader = data_loader, pairs_parser_factory =pairs_parser_factory , feature_extractor = feature_extractor, distance_metric = self.distance_metric) 
        mean_dis_false, mean_dis_true, mean_acc, mean_tpr, mean_fpr ,_, best_thres = evaluator_dataset.eval(self.model)
        
        metric_eval = "mean consine of pairs label" if self.distance_metric ==1 else "mean distance of pairs label"
        
        pretty_tabel = PrettyTable([metric_eval + ' false', metric_eval + ' true', "mean accuracy", "mean tpr", "mean fpr" , "time processing", "best threshold", "backbone type" ])
        pretty_tabel.add_row((mean_dis_false,  mean_dis_true,    mean_acc,    mean_tpr,    mean_fpr , time.time() - t ,best_thres, self.conf.backbone_type ))
        print(pretty_tabel)
        for row in pretty_tabel: 
            f.write(row.get_string())
        f.close()