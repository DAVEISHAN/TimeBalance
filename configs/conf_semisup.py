import numpy as np
import math

############ dataloader realated params
num_workers = 8
batch_size = 16#8

data_percentage = 1.0
v_batch_size = 24
num_modes = 10 
cropping_facs = [0.8]
fix_skip = 2
sr_ratio = 4


############ model input params
dataset = 'ucf101'

num_frames = 8
reso_h = 224 #112
reso_w = 224 #112


ori_reso_h = 240
ori_reso_w = 320

RGB = True #BEWARE OF THIS
# normalize = False#True
normalize = False #True #True




###### Training optimization related params
learning_rate = 1e-4 #1e-5
num_epochs = 150

scheduler_patience = 1
warmup = True
warmup_array = list(np.linspace(0.01,1, 10) + 1e-9) #[0.001, 0.1, 0.3, 0.5, 1.0]
val_freq = 3
opt_type = 'adam'
# opt_type = 'sgd'
# opt_type = 'adamW'



############## model related params
backbone = 'R3D50' 

frozen_bb = False
frozen_bn = False
kin_pretrained = False
num_classes = 102

pretrained_checkpoint = None 

checkpoint_inv = 'path/to/invariant_teacher/checkpoint' 
checkpoint_dist = 'path/to/distinctive_teacher/checkpoint'

if pretrained_checkpoint is not None:
    pretrained_checkpoint = pretrained_checkpoint.replace('-symlink', '')

linear = False
lr_scheduler = 'loss_based' # loss_based, cosine, patience_based
val_array = [0, 50, 60, 70] +[x*2 for x in range(40,num_epochs)]
min_crop_factor_training = 0.4
two_random_erase = True #default is True
aspect_ratio_aug = False #default is False

ssl_method = 'tclr_ssl' #could be 'tclr_r3d50'

checkpoint_tclr_checkpoint = 'path/to/tclr/checkpoint'
checkpoint_ssl = '' 

teacher_supervision = 'l2' #kld or l2
supervised_weight, labeled_teacher_loss_weight, unlabeled_teacher_loss_weight  = 1, 50, 100