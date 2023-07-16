import torch.nn as nn
import torch
from pytorchvideo.models.hub import slow_r50
from r3d import r3d_18
from r3d_classifier import r3d_18_classifier
from mlp import mlp_r50
from torch.cuda.amp import autocast

class backbone_wrapper(nn.Module):
    def __init__(self, retrieval = False):
        super(backbone_wrapper, self).__init__()
        self.retrieval = retrieval
        self.r3d50 = slow_r50(pretrained=False)
        self.r3d50.blocks[5] = nn.Identity()
        self.st_pooling = nn.AdaptiveAvgPool3d((4, 1,1))

    def forward(self,x):
        with autocast():
            if not self.retrieval:
                x, clip_type = x
            x = self.r3d50(x)
            x = self.st_pooling(x)
            if self.retrieval:
                x = x.squeeze(-1).squeeze(-1)
            return x, clip_type

def build_r3d50_mlp():
    f = backbone_wrapper()
    g = mlp_r50()
    model = nn.Sequential(f,g)
    return model

def load_r3d50_mlp(saved_model_file, avoid_layer4 = False, avoid_layer3 = False):
    model = build_r3d50_mlp()
    pretrained = torch.load(saved_model_file)
    pretrained_kvpair = pretrained['state_dict']
    model_kvpair = model.state_dict()

    for layer_name, weights in pretrained_kvpair.items():
        layer_name = layer_name.replace('module.','')
        if avoid_layer4 and ('blocks.4.r' in layer_name):
            print('avoided block4')
            continue
        if avoid_layer3 and ('blocks.3.r' in layer_name):
            print('avoided block 3')
            continue
        model_kvpair[layer_name] = weights  

    model.load_state_dict(model_kvpair, strict=True)
    print(f'{saved_model_file} loaded successfully')
    return model 

def build_r3d_classifier(arch = 'R3D18', num_classes = 102, kin_pretrained = False, self_pretrained = True, saved_model_file = None, retrieval = False):
    if arch == 'R3D18':
        model = r3d_18_classifier(pretrained = kin_pretrained, progress = False)
        model.layer4[0].conv1[0] = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(2, 1, 1),dilation = (2,1,1), bias=False)
        model.layer4[0].downsample[0] = nn.Conv3d(256, 512, kernel_size = (1, 1, 1), stride = (1, 2, 2), bias=False)
        model.fc = nn.Linear(512, num_classes)
        if retrieval:
            model.fc = nn.Identity()
    elif arch == 'R3D50':
        if retrieval:
            model = backbone_wrapper(retrieval = True)
        else:
            model = slow_r50(pretrained=False)
            model.blocks[5].proj = nn.Linear(2048, num_classes, bias= True)
    return model 

def load_r3d_classifier(arch = 'R3D18', num_classes = 102, saved_model_file = None):
    if arch == 'R3D18':
        model = r3d_18_classifier(pretrained = False, progress = False)
        model.layer4[0].conv1[0] = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(2, 1, 1),dilation = (2,1,1), bias=False)
        model.layer4[0].downsample[0] = nn.Conv3d(256, 512, kernel_size = (1, 1, 1), stride = (1, 2, 2), bias=False)
        model.fc = nn.Linear(512, num_classes)
    elif arch == 'R3D50':
        model = slow_r50(pretrained=False)
        model.blocks[5].proj = nn.Linear(2048, num_classes, bias= True)
    pretrained = torch.load(saved_model_file, map_location = 'cpu')
    pretrained_kvpair = pretrained['state_dict']
    model_kvpair = model.state_dict()
    for layer_name, weights in pretrained_kvpair.items():
        layer_name = layer_name.replace('module.0.','')
        model_kvpair[layer_name] = weights   
    model.load_state_dict(model_kvpair, strict=True)
    print(f'model {saved_model_file} loaded successsfully!')
    return model 

def build_r3d_backbone():
    model = r3d_18(pretrained = False, progress = False)
    model.layer4[0].conv1[0] = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(2, 1, 1),dilation = (2,1,1), bias=False)
    model.layer4[0].downsample[0] = nn.Conv3d(256, 512, kernel_size = (1, 1, 1), stride = (1, 2, 2), bias=False)
    return model

def load_r3d_backbone():
    model = build_r3d_backbone()
    pretrained = torch.load(saved_model_file)
    pretrained_kvpair = pretrained['state_dict']
    model_kvpair = model.state_dict()
    for layer_name, weights in pretrained_kvpair.items():
        layer_name = layer_name.replace('module.','')
        model_kvpair[layer_name] = weights  
    model.load_state_dict(model_kvpair, strict=True)
    print(f'{saved_model_file} loaded successfully')
    return model
