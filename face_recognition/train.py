
import os
import sys
import torch
import shutil
import argparse
import logging as logger
from tqdm import tqdm 
from sklearn import metrics

from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

sys.path.append('../../')
from utils.AverageMeter import AverageMeter
from dataset_utils.train_dataset import ImageDataset
from dataset_utils.evaluate_dataset import get_evaluate_dataset_and_loader
from dataset_utils.evaluate_helpers import *
from backbone.backbone_def import BackboneFactory
from losses.loss_def import LossFactory
from datetime import datetime, timedelta
import time
import config 
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# logger.basicConfig(level=logger.INFO, 
#                    format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
#                    datefmt='%Y-%m-%d %H:%M:%S')


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

    def forward(self, data, label):
        feat = self.backbone.forward(data)
        pred = self.loss.forward(feat, label)
        return pred
    def feed_emb(self, data): 
        return self.backbone.forward(data) 
    
class FaceTrainer(object): 
    def __init__(self, conf, inference=False):
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
        self.log_file_path = os.path.join(conf.log_dir, 'history_training_log.txt')
        # Load data
        dataset = ImageDataset(conf.data_root, conf.image_shape)
        self.num_class = conf.num_class = dataset.__num_class__()
        self.data_loader = DataLoader(dataset, conf.batch_size, True, num_workers = 4, drop_last= True)
        # Define criterion loss 
        self.criterion = torch.nn.CrossEntropyLoss().to(conf.device)
        # Load backbone 
        backbone_factory = BackboneFactory(conf.backbone_type, conf)    
        # Load losses
        loss_factory = LossFactory(conf.loss_type, conf)
        # Load models
        self.model = FaceModel(backbone_factory, loss_factory)
        print('Generated models {} deep layer {} type loss {} done !. \n'
              .format(conf.backbone_type,conf.model_parameter[conf.backbone_type]['depth'] ,conf.loss_type))
        # load pretrained 
        
        if conf.device.type != "cpu":    
            self.model = torch.nn.DataParallel(self.model).cuda()
        # init optimizer lr_schedule and loss_meter     
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.SGD(parameters, lr = conf.lr, momentum = conf.momentum, weight_decay = 1e-4)
        self.lr_schedule = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones = conf.milestones, gamma = 0.1)
        self.loss_meter = AverageMeter()
        if conf.resume: 
            self.load_state(conf, True)
        # load evaluate dataset 
        self.evaluate_dataset =  self.evaluate_loader(conf)
    
    def evaluate_loader(self, conf): 
        validation_paths_dic = {
                    "LFW" : os.path.join(conf.evaluate_dataset_root, 'lfw_112'),
                    "CALFW" : os.path.join(conf.evaluate_dataset_root, 'calfw_112'),
                    "CPLFW" : os.path.join(conf.evaluate_dataset_root, 'cplfw_112'),
                    "CFP_FF" : os.path.join(conf.evaluate_dataset_root, 'cfp_112'),
                    "CFP_FP" : os.path.join(conf.evaluate_dataset_root, 'cfp_112')}
        validation_data_dic = {}
        for val_type in conf.validations:
            
            self.print_and_log('Init dataset and loader for validation type: {}'.format(val_type))
            print('Init dataset and loader for validation type: {}'.format(val_type))
            
            dataset, loader = get_evaluate_dataset_and_loader(root_dir=validation_paths_dic[val_type], 
                                                                    type=val_type, 
                                                                    num_workers=conf.num_workers, 
                                                                    input_size=conf.image_shape, 
                                                                    batch_size=conf.evaluate_batch_size)
            validation_data_dic[val_type+'_dataset'] = dataset
            validation_data_dic[val_type+'_loader'] = loader
        return validation_data_dic 

    def load_state(self, conf, load_optimizer =False): 
        if os.path.exists(conf.pretrain_model_path): 
            state_dict = torch.load(conf.pretrain_model_path, map_location = conf.device )
            self.model.load_state_dict(state_dict['state_dict'])
            if load_optimizer: 
                self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            print('load model epoch {} and batch_id {} and load optimizer {} \n'.format(state_dict['epoch'], state_dict['batch_id'],load_optimizer))
        else: 
            print('pretrained model path not exist !')

    def save_state(self, saved_name, epoch, batch_id, conf): 
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'batch_id': batch_id
        }
        torch.save(state, os.path.join(conf.out_dir, saved_name))

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
        
        self.model.train()
        for epoch in range(conf.epoches):
            batch_idx = 0 
            print('\n')
            for (images, labels) in tqdm(self.data_loader, desc='epoch {} started'.format(epoch)):
            # for (images, labels) in self.data_loader:
                images = images.to(conf.device)
                labels = labels.to(conf.device)
                labels = labels.squeeze()
                outputs = self.model.forward(images, labels)
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.loss_meter.update(loss.item(), images.shape[0])
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
                    
                if (batch_idx + 1) % conf.save_freq == 0:
                    saved_name = 'Epoch_%d_batch_%d.pt' % (epoch, batch_idx)
                    self.save_state(saved_name, epoch, batch_idx, conf)
                batch_idx +=1
                self.step_loop +=1
            print('epoch {}'.format(epoch))    
            if epoch % conf.evaluate_every_epoch == 0: 
                """
                code evaluate dataset  
                """
                print('evaluating model ....')    
                for val_type in conf.validations:
                    dataset = self.evaluate_dataset[val_type+'_dataset']
                    loader = self.evaluate_dataset[val_type+'_loader']
                    self.model.eval()
                    t = time.time()
                    print('\n\nRunnning forward pass on {} images'.format(val_type))

                    tpr, fpr, accuracy, val, val_std, far = evaluate_forward_pass(self.model, 
                                                                                loader, 
                                                                                dataset, 
                                                                                conf.feat_dim, 
                                                                                conf.device,
                                                                                lfw_nrof_folds=conf.evaluate_nrof_folds, 
                                                                                distance_metric= conf.distance_metric, 
                                                                                subtract_mean=conf.evaluate_subtract_mean)

                    self.print_and_log('\nEpoch: '+str(epoch))
                    self.print_and_log('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
                    self.print_and_log('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

                    auc = metrics.auc(fpr, tpr)
                    self.print_and_log('Area Under Curve (AUC): %1.3f' % auc)

                    time_for_val = int(time.time() - t)
                    self.print_and_log('Total time for {} evaluation: {}'.format(val_type, timedelta(seconds=time_for_val)))
                    print("\n")
                        
                    self.writer.add_scalar(val_type +"_accuracy", np.mean(accuracy), epoch)
                            
            # save final every epoch 
            saved_name = 'Final_epoch_%d.pt' % epoch
            self.save_state(saved_name, epoch, 0 , conf)
            self.lr_schedule.step()
        self.writer.close()                        


if __name__ == '__main__':
    config.milestones = [int(num) for num in config.step.split(',')]
    logger.info('Start optimization.')
    logger.info(config)
    learner = FaceTrainer(config)
    learner.train(config)
    logger.info('Optimization done!')
