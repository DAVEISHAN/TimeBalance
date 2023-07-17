import numpy as np
import math

############ Dataset related hyperparameters
dataset = 'ucf101'
num_workers = 8
batch_size = 16 #8
data_percentage = 1.0
fix_skip = 2
sr_ratio = 4
num_frames = 8
reso_h = 224 #112
reso_w = 224 #112
ori_reso_h = 240
ori_reso_w = 320
RGB = True #BEWARE OF THIS
normalize = False #True #True
v_batch_size = 24
num_modes = 10
cropping_facs = [0.8]#[0.8, 1.0]
min_crop_factor_training = 0.4
two_random_erase = True #default is True
aspect_ratio_aug = False #default is False
dl_mode = 'ucf_10p' # could be '20p'

############ Model related hyperparameters
backbone = 'R3D50'
frozen_bb = False
frozen_bn = False
kin_pretrained = False
num_classes = 102
linear = False
pretrained_checkpoint = '/path/to/pretrained/teacher/checkpoint'
if pretrained_checkpoint is not None:
    pretrained_checkpoint = pretrained_checkpoint.replace('-symlink', '')

############ Training optimization related hyperparameters
learning_rate = 1e-4
num_epochs = 150
scheduler_patience = 1
warmup = True
warmup_array = list(np.linspace(0.01,1, 10) + 1e-9) #[0.001, 0.1, 0.3, 0.5, 1.0]
val_freq = 3
opt_type = 'adam'
# opt_type = 'sgd'
# opt_type = 'adamW'
lr_scheduler = 'loss_based' # loss_based, cosine, patience_based
val_array = [0, 10, 20,30, 40, 50] +[x*2 for x in range(40,num_epochs)]
