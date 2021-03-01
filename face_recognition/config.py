import torch 
# invironment training cuda:0 or cpu 
# device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")

data_root = '/home/minglee/Documents/aiProjects/git_clone/Dataset/VN-celeb' 
#help = "The root folder of training set."
train_file = '/export2/wangjun492/face_database/facex-zoo/private_file/train_data/deepglint/msceleb_deepglint_train_file.txt' 
#help = "The training file path."
backbone_type =  'ResNet'  
# ['ir', 'ir_se'], 'mode should be ir or ir_se' ,[50, 100, 152], 'num_layers should be 50,100, or 152'
# help = "Mobilefacenets, Resnet."  
loss_type = 'ArcFace' 
# help = "mv-softmax, arcface, npc-face."
lr = 0.1 
# help='The initial learning rate.'
out_dir = '/home/minglee/Documents/aiProjects/git_clone/trash/history/weights' 
# help = "The folder to save models."
epoches = 18 
# help = 'The training epoches.'
step = '10, 13, 16' 
# help = 'Step for lr.'
print_freq = 200 
# help = 'The print frequency for training state.'
save_freq = 3000 
# help = 'The save frequency for training state.'
batch_size = 2
# help='The training batch size over all gpus.'
momentum = 0.9 
# help = 'The momentum for sgd.'
log_dir = '/home/minglee/Documents/aiProjects/git_clone/trash/history/log' 
# help = 'The directory to save log.log'
tensorboardx_logdir = 'arc-resnet' 
# help = 'The directory to save tensorboardx logs'
pretrain_model = 'arc_resnet_epoch_8.pt'
# help = 'The path of pretrained model'
resume = False
# help = 'Whether to resume from a checkpoint.'
num_class = 72778
#number of class
feat_dim = 512
#shape of embedding
image_shape = (112,112)

# model type 
model_parameter = {'ResNet': 
                      {'depth': 50,   # 50,100, or 152'
                      'drop_ratio': 0.4, 
                      'net_mode': 'ir',  # ['ir', 'ir_se']
                      'feat_dim': feat_dim, 
                      'out_h': 7, 
                      'out_w': 7},
                  'MobileFaceNet': 
                      {'feat_dim': feat_dim, 
                      'out_h': 7, 
                      'out_w': 7 }
                      } 
  
# loss type 
loss_paramenter = {'ArcFace':
                        {'feat_dim': feat_dim,
                        'num_class': num_class,
                        'margin_arc': 0.35,
                        'margin_am': 0.0,
                        'scale': 32},
                    'AM-Softmax':
                        {'feat_dim': feat_dim,
                        'num_class': num_class,
                        'margin': 0.35,
                        'scale': 32},
                    'AdaCos':
                        {'feat_dim': feat_dim,
                        'num_class': num_class},
                    'AdaM-Softmax':
                            {'feat_dim': feat_dim,
                            'num_class': num_class,
                            'scale': 32,
                            'lamda': 70.0},
                    'MV-Softmax':
                            {'feat_dim': feat_dim,
                            'num_class': num_class,
                            'is_am': 1,
                            'margin': 0.35,
                            'mv_weight': 1.12,
                            'scale': 32}
                    }













