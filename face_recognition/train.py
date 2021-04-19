
import os
import sys
import torch
import shutil
import argparse
import logging as logger
from tqdm import tqdm 
from sklearn import metrics
import yaml

from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

sys.path.append('../../')
from utils.AverageMeter import AverageMeter
from backbone.backbone_def import BackboneFactory
from losses.loss_def import LossFactory
from datetime import datetime, timedelta

from prettytable import PrettyTable
from dataset_utils.evaluator_dataset import Evaluator
from dataset_utils.extractor_embedding import CommonExtractor 
from dataset_utils.pairs_parser import PairsParserFactory
from dataset_utils.train_dataset import ImageDataset, CommonTestDataset
from dataset_utils.module_eval import ModuleEval
from utils.model_loader import ModelLoader 

import time
import config 
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

logger.basicConfig(level=logger.INFO, 
                   format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')
class FaceModel(torch.nn.Module):
    """Define a traditional face model which contains a backbone and a head.
    
    Attributes:
        backbone(object): the backbone of face model.
        head(object): the head of face model.
    """
    def __init__(self, backbone_factory, loss_factory):
        """Init face model by backbone factorcy and head factory.
        
        conf:
            backbone_factory(object): produce a backbone according to conf files.
            head_factory(object): produce a head according to conf files.
        """
        super(FaceModel, self).__init__()
        self.backbone = backbone_factory.get_backbone()
        self.loss = loss_factory.get_loss()

    def forward(self, data, label, status = 'train'):
        
        feat = self.backbone.forward(data)
        if status == 'eval': 
            return feat
        pred = self.loss.forward(feat, label)

        return pred
    
class FaceTrainer(object): 
    def __init__(self, conf, inference=False):
        if not os.path.exists(conf.log_dir): 
            os.makedirs(conf.log_dir)
        self.log_file_path = os.path.join(conf.log_dir, 'history_training_log.txt')
        # Load backbone 
        backbone_factory = BackboneFactory(conf.backbone_type, conf.model_parameter[conf.backbone_type])    
        # Load losses
        loss_factory = LossFactory(conf.loss_type, conf.loss_parameter[conf.loss_type])
        # Load models
        self.model = FaceModel(backbone_factory, loss_factory)

        print(' Load backbone', conf.model_parameter[conf.backbone_type])
        print(' Load loss model', conf.loss_parameter[conf.loss_type])
        self.print_and_log(' Load backbone {}'.format(conf.model_parameter[conf.backbone_type]))
        self.print_and_log(' Load loss model {}'.format(conf.loss_parameter[conf.loss_type]))
        
        self.evaluator = ModuleEval(conf.root_eval_dataset,self.model,conf,gen_pair=True,status='train')

        if not inference: 
            self.step_loop = 0 
            # init tensorboard writer history and paramenters 
            if not os.path.exists(conf.out_dir):
                os.makedirs(conf.out_dir)
            if not os.path.exists(conf.log_dir):
                os.makedirs(conf.log_dir)
            tensorboardx_logdir = os.path.join(conf.log_dir, conf.tensorboardx_logdir)
            
            print('path of tensorboard: ', tensorboardx_logdir)
            self.writer = SummaryWriter(log_dir=tensorboardx_logdir)    
            # init history of train models 
            # Load data
            dataset = ImageDataset(conf.data_root, conf.image_shape)
            self.num_class = conf.num_class = dataset.__num_class__()
            self.data_loader = DataLoader(dataset, conf.batch_size, True, num_workers = 4, drop_last= True)
            # Define criterion loss 
            self.criterion = torch.nn.CrossEntropyLoss().to(conf.device)

            # init optimizer lr_schedule and loss_meter     
            parameters = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = optim.SGD(parameters, lr = conf.lr, momentum = conf.momentum, weight_decay = 1e-4)
            self.lr_schedule = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones = conf.milestones, gamma = 0.1)
            self.loss_meter = AverageMeter()
                

    def load_state(self, conf, load_optimizer =False): 
        status_model_load = torch.load(conf.pretrain_model, map_location=conf.device)
        state_dict = status_model_load['state_dict']
        state_dict['loss.weight'] = state_dict['head.weight']
        del state_dict['head.weight']
        self.model.load_state_dict(state_dict)
        if conf.device.type != 'cpu': 
            self.model = torch.nn.DataParallel(self.model).cuda()
        print('load pretrained done! ')
        self.print_and_log('load pretrained model path:{}'.format(conf.pretrain_model))
        self.print_and_log('epoch:{} batch_idx {}'.format(status_model_load['epoch'], status_model_load['batch_id']))
        self.print_and_log('load pretrained done! ')

    def get_lr(self):
        """Get the current learning rate from optimizer. 
        """
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
        
    def print_and_log(self, string_to_write):
        with open(self.log_file_path, "a") as log_file:
            # t = "[" + str(datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')) + "] " 
            # log_file.write(t + string_to_write + "\n")
            log_file.write(string_to_write + '\n')
    
    def train(self, conf):
        """Total training procedure.
        """
        if conf.reload_model:
            self.load_state(conf)

        self.model.train()
        self.print_and_log('started training process .........')
        for epoch in range(conf.epoches):
            
            batch_idx = 0 
            print('\n')
            self.print_and_log('started epoch:%d\n'%(epoch))
            for (images, labels) in tqdm(self.data_loader, desc='started epoch: {}'.format(epoch)):
                
                images = images.to(conf.device)
                labels = labels.to(conf.device)
                labels = labels.squeeze()
                pred = self.model.forward(images, labels)
                loss = self.criterion(pred, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.loss_meter.update(loss.item(), images.shape[0])

                # save log parametor
                if batch_idx % conf.print_freq == 0:
                    loss_avg = self.loss_meter.avg
                    lr = self.get_lr()
                    log = 'Epoch %d, iter %d/%d, lr %f, loss %f'%(epoch, batch_idx, len(self.data_loader), lr, loss_avg)
                    self.print_and_log(log)
                    self.writer.add_scalar('Train_loss', loss.item(), self.step_loop)
                    log = 'Train loss %f step %d'%(loss.item(), self.step_loop)
                    self.print_and_log(log)
                    self.writer.add_scalar('smooth_loss', loss_avg, self.step_loop)
                    self.writer.add_scalar('Train_lr', lr, self.step_loop)
                    log = 'Train_lr %f step %d'%(lr, self.step_loop)
                    self.print_and_log(log)
                    self.loss_meter.reset()
                # test model 
                if (batch_idx + 1) % conf.eval_by_batch_idx == 0 and batch_idx !=0: 
                    print('evaluating model in epoch: {} batch_id {}'.format(epoch,batch_idx))  
                    self.print_and_log('evaluating model in epoch: {} batch_id {}'.format(epoch,batch_idx))  
                    self.evaluator.eval()
                   
                # save batch_idx model
                if (batch_idx + 1) % conf.save_freq == 0:
                    saved_name = 'Epoch_%d_batch_%d.pt' % (epoch, batch_idx)
                    state = { 'state_dict': self.model.state_dict(),
                                'epoch': epoch,
                                'batch_id': batch_idx}
                    torch.save(state, os.path.join(conf.out_dir, saved_name))
                batch_idx +=1
            
            self.step_loop +=1

            # save model by end of single epoch
            saved_name = 'Epoch_%d.pt' % epoch
            state = {'state_dict': self.model.state_dict(), 
                        'epoch': epoch, 
                        'batch_id': batch_idx}

            torch.save(state, os.path.join(conf.out_dir, saved_name))
            self.print_and_log('save model name{}'.format(saved_name))
            self.lr_schedule.step()
        self.print_and_log('end training process .........')
        self.writer.close()                        

     
if __name__ == '__main__':
    config.milestones = [int(num) for num in config.step.split(',')]
    learner = FaceTrainer(config)
    learner.train(config)