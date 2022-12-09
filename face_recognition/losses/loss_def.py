
import sys
sys.path.append('../../')
from losses.AM_Softmax import AM_Softmax
from losses.ArcFace import ArcFace

class LossFactory:
    """Factory to produce head according to the head_conf.yaml
    
    Attributes:
        loss_type(str): which head will be produce.
    """
    def __init__(self, loss_type, loss_parameter):
        self.loss_type = loss_type
        self.loss_paramenter = loss_parameter
    def get_loss(self):
        if self.loss_type == 'AM-Softmax':
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
        else:
            pass
        return loss
