import torch 
# invironment training cuda:0 or cpu 
# device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

data_root = '/home/minglee/Documents/aiProjects/RepoGithub/Dataset/VN-celeb' 
#help = "The root folder of training set."
backbone_type =  'ResNet'  
# ['ir', 'ir_se'], 'mode should be ir or ir_se' ,[50, 100, 152], 'num_layers should be 50,100, or 152'
# help = "Mobilefacenets, Resnet."  
loss_type = 'ArcFace' 
# help = "mv-softmax, arcface, npc-face."
lr = 0.1 
# help='The initial learning rate.'
out_dir = '/home/minglee/Documents/aiProjects/RepoGithub/F-Vision/face_recognition/trash/Output_models/history/weights' 
# help = "The folder to save models."
epoches = 4 
# help = 'The training epoches.'
step = '10, 13, 16' 
# help = 'Step for lr.'
print_freq = 2 
# help = 'The print frequency for training state.'
save_freq = 100 
# help = 'The save frequency for training state.'
batch_size = 2
# help='The training batch size over all gpus.'
momentum = 0.9 
# help = 'The momentum for sgd.'
log_dir = '/home/minglee/Documents/aiProjects/RepoGithub/F-Vision/face_recognition/trash/Output_models/history/log' 
# help = 'The directory to save log.log'
tensorboardx_logdir = 'tensorboard' 
# help = 'The directory to save tensorboardx logs'
pretrain_model_path = '/home/minglee/Documents/aiProjects/RepoGithub/F-Vision/face_recognition/trash/Output_models/history/weights/Final_epoch_1.pt'
# help = 'The path of pretrained model'
resume = True
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
                        'scale': 32}
                    }













