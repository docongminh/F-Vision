
import sys
import yaml
sys.path.append('../../')
from losses.AdaCos import AdaCos
from losses.AdaM_Softmax import Adam_Softmax
from losses.AM_Softmax import AM_Softmax
from losses.ArcFace import ArcFace
from losses.MV_Softmax import MV_Softmax

class LossFactory:
    """Factory to produce head according to the head_conf.yaml
    
    Attributes:
        loss_type(str): which head will be produce.
    """
    def __init__(self, loss_type, conf):
        self.loss_type = loss_type
        self.loss_paramenter = conf.loss_paramenter[loss_type]

    def get_head(self):
        if self.loss_type == 'AdaCos':
            feat_dim = self.loss_paramenter['feat_dim'] # dimension of the output features, e.g. 512 
            num_class = self.loss_paramenter['num_class'] # number of classes in the training set.
            loss = AdaCos(feat_dim, num_class)
        elif self.loss_type == 'AdaM-Softmax':
            feat_dim = self.loss_paramenter['feat_dim'] # dimension of the output features, e.g. 512 
            num_class = self.loss_paramenter['num_class'] # number of classes in training set.
            scale = self.loss_paramenter['scale'] # the scaling factor for cosine values.
            lamda = self.loss_paramenter['lamda'] # controls the strength of the margin constraint Lm.
            loss = Adam_Softmax(feat_dim, num_class, scale, lamda)
        elif self.loss_type == 'AM-Softmax':
            feat_dim = self.loss_paramenter['feat_dim'] # dimension of the output features, e.g. 512 
            num_class = self.loss_paramenter['num_class'] # number of classes in the training set.
            margin = self.loss_paramenter['margin'] # cos_theta - margin
            scale = self.loss_paramenter['scale'] # the scaling factor for cosine values.
            loss = AM_Softmax(feat_dim, num_class, margin, scale)
        elif self.loss_type == 'ArcFace':
            feat_dim = self.loss_paramenter['feat_dim'] # dimension of the output features, e.g. 512 
            num_class = self.loss_paramenter['num_class'] # number of classes in the training set.
            margin_arc = self.loss_paramenter['margin_arc'] # cos(theta + margin_arc).
            margin_am = self.loss_paramenter['margin_am'] # cos_theta - margin_am.
            scale = self.loss_paramenter['scale'] # the scaling factor for cosine values.
            loss = ArcFace(feat_dim, num_class, margin_arc, margin_am, scale)
        elif self.loss_type == 'MV-Softmax':
            feat_dim = self.loss_paramenter['feat_dim'] # dimension of the output features, e.g. 512 
            num_class = self.loss_paramenter['num_class'] # number of classes in the training set.
            is_am = self.loss_paramenter['is_am'] # am-softmax for positive samples.
            margin = self.loss_paramenter['margin'] # margin for positive samples.
            mv_weight = self.loss_paramenter['mv_weight'] # weight for hard negtive samples.
            scale = self.loss_paramenter['scale'] # the scaling factor for cosine values.
            loss = MV_Softmax(feat_dim, num_class, is_am, margin, mv_weight, scale)
        else:
            pass
        return loss
