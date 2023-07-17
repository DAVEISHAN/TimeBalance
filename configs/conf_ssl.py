import numpy as np
import math

############ Input related hyperparameters
dataset = 'ucf101'
num_workers = 8
batch_size = 40 
data_percentage = 1.0
fix_skip = 2
sr_ratio = 4
num_frames = 16
reso_h = 224 #112
reso_w = 224 #112
RGB = True #BEWARE OF THIS
normalize = False #True #True

augmentations_dict = {
    "cropping_fac": 0.5,
    "blur": False,  
    # "blur_kernelSize":
    # "blur_sigma": 
    # "random_eraseSize_fac":     
}

############ Model related hyperparameters
backbone = 'R3D50'
frozen_bb = False
frozen_bn = False
kin_pretrained = False
num_classes = 102
num_dims = 512
linear = False
pretrained_checkpoint = None 
if pretrained_checkpoint is not None:
    pretrained_checkpoint = pretrained_checkpoint.replace('-symlink', '')
ic_weight, tcl_weight, tcg_weight = 1,1,1

############ Training optimization related hyperparameters
learning_rate = 5e-3 #1e-3 #1e-5
weight_decay = 1e-9
num_epochs = 1000
temperature = 0.1
scheduler_patience = 5
drop_factor = 2
warmup = True
warmup_array = list(np.linspace(0.01,1, 10) + 1e-9) #[0.001, 0.1, 0.3, 0.5, 1.0]
val_freq = 3
opt_type = 'adam' 
# opt_type = 'sgd'
# opt_type = 'adamW'
lr_scheduler = 'loss_based' # loss_based, cosine, patience_based
