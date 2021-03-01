
import os
import sys
import config
import shutil
import argparse
import logging as logger
import config

import torch
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

sys.path.append('../../')
from utils.AverageMeter import AverageMeter
from dataset_utils.train_dataset import ImageDataset
from backbone.backbone_def import BackboneFactory
from losses.loss_def import LossFactory

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
        
        config:
            backbone_factory(object): produce a backbone according to config files.
            head_factory(object): produce a head according to config files.
        """
        super(FaceModel, self).__init__()
        self.backbone = backbone_factory.get_backbone()
        self.loss = loss_factory.get_head()

    def forward(self, data, label):
        feat = self.backbone.forward(data)
        pred = self.loss.forward(feat, label)
        return pred

def get_lr(optimizer):
    """Get the current learning rate from optimizer. 
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_one_epoch(data_loader, model, optimizer, criterion, cur_epoch, loss_meter, conf):
    """Tain one epoch by traditional training.
    """
    for batch_idx, (images, labels) in enumerate(data_loader):
        images = images.to(conf.device)
        labels = labels.to(conf.device)
        labels = labels.squeeze()
        if conf.loss_type == 'AdaM-Softmax':
            outputs, lamda_lm = model.forward(images, labels)
            lamda_lm = torch.mean(lamda_lm)
            loss = criterion(outputs, labels) + lamda_lm
        else:
            outputs = model.forward(images, labels)
            loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item(), images.shape[0])
        if batch_idx % conf.print_freq == 0:
            loss_avg = loss_meter.avg
            lr = get_lr(optimizer)
            logger.info('Epoch %d, iter %d/%d, lr %f, loss %f' % 
                        (cur_epoch, batch_idx, len(data_loader), lr, loss_avg))
            global_batch_idx = cur_epoch * len(data_loader) + batch_idx
            conf.writer.add_scalar('Train_loss', loss_avg, global_batch_idx)
            conf.writer.add_scalar('Train_lr', lr, global_batch_idx)
            loss_meter.reset()
        if (batch_idx + 1) % conf.save_freq == 0:
            saved_name = 'Epoch_%d_batch_%d.pt' % (cur_epoch, batch_idx)
            state = {
                'state_dict': model.module.state_dict(),
                'epoch': cur_epoch,
                'batch_id': batch_idx
            }
            torch.save(state, os.path.join(conf.out_dir, saved_name))
            logger.info('Save checkpoint %s to disk.' % saved_name)
    saved_name = 'Epoch_%d.pt' % cur_epoch
    state = {'state_dict': model.module.state_dict(), 
             'epoch': cur_epoch, 'batch_id': batch_idx}
    torch.save(state, os.path.join(conf.out_dir, saved_name))
    logger.info('Save checkpoint %s to disk...' % saved_name)

def train(conf):
    """Total training procedure.
    """
    dataset = ImageDataset(conf.data_root, conf.image_shape)
    config.num_class = dataset.__num_class__()
    data_loader = DataLoader(dataset, conf.batch_size, True, num_workers = 4)
    criterion = torch.nn.CrossEntropyLoss().to(conf.device)
    backbone_factory = BackboneFactory(conf.backbone_type, conf)    
    loss_factory = LossFactory(conf.loss_type, conf)
    model = FaceModel(backbone_factory, loss_factory)
    ori_epoch = 0
    if conf.resume:
        ori_epoch = torch.load(config.pretrain_model)['epoch'] + 1
        state_dict = torch.load(config.pretrain_model)['state_dict']
        model.load_state_dict(state_dict)
    model = torch.nn.DataParallel(model).cuda()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(parameters, lr = conf.lr, 
                          momentum = conf.momentum, weight_decay = 1e-4)
    lr_schedule = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones = conf.milestones, gamma = 0.1)
    loss_meter = AverageMeter()
    model.train()
    for epoch in range(ori_epoch, conf.epoches):
        train_one_epoch(data_loader, model, optimizer, 
                        criterion, epoch, loss_meter, conf)
        lr_schedule.step()                        

if __name__ == '__main__':
    
    config.milestones = [int(num) for num in config.step.split(',')]
    if not os.path.exists(config.out_dir):
        os.makedirs(config.out_dir)
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    tensorboardx_logdir = os.path.join(config.log_dir, config.tensorboardx_logdir)
    if os.path.exists(tensorboardx_logdir):
        shutil.rmtree(tensorboardx_logdir)
    writer = SummaryWriter(log_dir=tensorboardx_logdir)
    config.writer = writer
    
    logger.info('Start optimization.')
    logger.info(config)
    train(config)
    logger.info('Optimization done!')
