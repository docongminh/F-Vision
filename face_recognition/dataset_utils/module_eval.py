import torch 
import os 
import numpy as np 
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
    def __init__(self, root_dir, text_file_path, text_label_path, num_of_pair ):
        self.nofpair = int(num_of_pair) 
        self.root_dir = root_dir 
        self.text_file_path = text_file_path
        self.text_label_path = text_label_path
        if not os.path.exists(self.text_file_path): 
            print('generate pair pairse file \n')
            self.num_of_pair_false()
            self.num_of_pair_true()
        if not os.path.exists(self.text_label_path):
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
    
        f = open(self.text_file_path, "a")

       
        folder_img = os.listdir(self.root_dir)
        sub_first = os.listdir(self.root_dir + '/' + folder_img[0])
        cnt = 0
        for i, sub_folder_second in enumerate(folder_img): 
            sub_folder = os.listdir(self.root_dir + '/' + sub_folder_second)
            if i != 0: 
                for img_first in sub_first: 
                    for img_second in sub_folder: 
                        f.write(str(folder_img[0]+ '/' + img_first) + " " + str(sub_folder_second + '/' + img_second) + " " + str(0) + "\n")
                        cnt += 1
                        if cnt >= int(self.nofpair//2): 
                            return 

        f.close()

    def num_of_pair_true(self): 
        f = open(self.text_file_path, "a")
        
        folder_img = os.listdir(self.root_dir) 
        cnt = 0 
        for sub_folder in folder_img: 
            sub_list = os.listdir(self.root_dir + '/' + sub_folder)
            if len(sub_list) < 3: 
                continue
            for idxf, img_first in enumerate(sub_list): 
                for idxs, img_second in enumerate(sub_list): 
                    if idxs > idxf: 
                        f.write(str(sub_folder +'/'+img_first) + " " + str(sub_folder + '/' + img_second) + " " + str(1) + "\n")
                        cnt +=1 
                        if cnt >= int(self.nofpair//2): 
                            return 

        f.close() 


class ModuleEval():  
    """
    """ 
    def __init__(self, data_path, model, conf, gen_pair=True): 
        self.model = model 
        if torch.cuda.is_available(): 
            model = torch.nn.DataParallel(self.model).cuda()
        self.conf = conf
        self.data_path = data_path
        self.text_file_path = self.data_path + '/' + 'pairs_file_path.txt'
        self.text_label_path = self.data_path + '/' + 'image_list_file_path.txt'
        if gen_pair:
            GenPairImage(self.data_path, self.text_file_path, self.text_label_path, conf.num_of_pair)


    def eval(self): 
        inf_list = []
        feature_extractor = CommonExtractor(self.conf.device) 


        print("load eval dataset ! \n")    
        t = time.time()
        pairs_file =            self.text_file_path
        cropped_foler_img =     self.data_path
        image_list_label_path = self.text_label_path 

        pairs_parser_factory = PairsParserFactory(pairs_file)
        data_loader = DataLoader(CommonTestDataset(cropped_foler_img, image_list_label_path, False),
                        batch_size= self.conf.evaluate_batch_size, num_workers = self.conf.num_workers, shuffle=False)
        feature_extractor = CommonExtractor(self.conf.device)
        evaluator_dataset = Evaluator(data_loader, pairs_parser_factory, feature_extractor) 
        mean_dis_false, mean_dis_true, mean_acc, mean_tpr, mean_fpr ,std, best_thres = evaluator_dataset.eval(self.model)

        inf_list.append((mean_dis_false, mean_dis_true, mean_acc, mean_tpr, mean_fpr, std, time.time()-t, best_thres,'VN-celeb'))

        pretty_tabel = PrettyTable(['mean distance false','mean distance true',"mean accuracy","mean tpr","mean fpr" , "standard error", "time processing", "best threshold" ,"dataset type"])
        for row in inf_list:
            pretty_tabel.add_row(row)
        print(pretty_tabel)       



# if __name__=='__main__': 
#     root_dir = conf.data_root
#     text_file_path = conf.data_root + '/' + 'pairs_file_path.txt'
#     text_label_path = conf.data_root + '/' + 'image_list_file_path.txt'
    

#     backbone_parametor = conf.model_parameter[conf.backbone_type]  
#     backbone_factory = BackboneFactory(conf.backbone_type, backbone_parametor) 
#     model_loader = ModelLoader(backbone_factory, conf.device) 
#     model = model_loader.load_model(conf.pretrained_paths)
#     print('load model done !')
#     print('backbone: {}: model_parametor:{}'.format(conf.backbone_type, backbone_parametor))
#     evaluate_dataset = ModuleEval(conf.data_root, model, conf, True)
#     evaluate_dataset.eval()