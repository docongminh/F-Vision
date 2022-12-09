
import sys
sys.path.append('../../')
from backbone.ResNets import Resnet
from backbone.MobileFaceNets import MobileFaceNet

class BackboneFactory:
    """Factory to produce backbone according the backbone_conf.yaml.
    
    Attributes:
        backbone_type(str): which backbone will produce.
        backbone_param(dict):  parsed params and it's value. 
    """
    def __init__(self, backbone_type, model_parameter):
        self.backbone_type = backbone_type
        self.model_parameter = model_parameter

    def get_backbone(self):
        if self.backbone_type == 'MobileFaceNet':
            feat_dim = self.model_parameter['feat_dim'] # dimension of the output features, e.g. 512.
            out_h = self.model_parameter['out_h'] # height of the feature map before the final features.
            out_w = self.model_parameter['out_w'] # width of the feature map before the final features.
            backbone = MobileFaceNet(feat_dim, out_h, out_w)
        elif self.backbone_type == 'ResNet':
            depth = self.model_parameter['depth'] # depth of the ResNet, e.g. 50, 100, 152.
            drop_ratio = self.model_parameter['drop_ratio'] # drop out ratio.
            net_mode = self.model_parameter['net_mode'] # 'ir' for improved by resnt, 'ir_se' for SE-ResNet.
            feat_dim = self.model_parameter['feat_dim'] # dimension of the output features, e.g. 512.
            out_h = self.model_parameter['out_h'] # height of the feature map before the final features.
            out_w = self.model_parameter['out_w'] # width of the feature map before the final features.
            backbone = Resnet(depth, drop_ratio, net_mode, feat_dim, out_h, out_w)
        else:
            pass
        return backbone
