import torch
import torch.nn as nn
from torch.cuda.amp import autocast

class mlp(nn.Module):
    def __init__(self, final_embedding_size = 128, use_normalization = True):
        super(mlp, self).__init__()
        self.final_embedding_size = final_embedding_size
        self.use_normalization = use_normalization
        self.fc1 = nn.Linear(512,512, bias = True)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, self.final_embedding_size, bias = False)
        self.temp_avg = nn.AdaptiveAvgPool3d((1,None,None))

    def forward(self, x):
        with autocast():
            x, clip_type = x
            if clip_type == 'd':
                x = self.temp_avg(x)
                x = x.flatten(1)
                x = self.relu(self.bn1(self.fc1(x)))
                x = nn.functional.normalize(self.bn2(self.fc2(x)), p=2, dim=1)
                return x
            elif clip_type == 's':
                gsr = self.temp_avg(x)
                gsr = gsr.flatten(1)
                gsr = self.relu(self.bn1(self.fc1(gsr)))
                gsr = nn.functional.normalize(self.bn2(self.fc2(gsr)), p=2, dim=1)
                x1, x2, x3, x4 = [nn.functional.normalize(self.bn2(self.fc2(
                                    self.relu(self.bn1(self.fc1(x[:,:,i,:,:].flatten(1))))))) for i in range(4)]
                return gsr, x1, x2, x3, x4
            else:
                return None, None, None, None, None

class mlp_r50(nn.Module):
    def __init__(self, final_embedding_size = 128, use_normalization = True):
        super(mlp_r50, self).__init__()
        self.final_embedding_size = final_embedding_size
        self.use_normalization = use_normalization
        self.fc1 = nn.Linear(2048,final_embedding_size, bias = True)
        self.bn1 = nn.BatchNorm1d(final_embedding_size)
        self.bn2 = nn.BatchNorm1d(final_embedding_size)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(final_embedding_size, self.final_embedding_size, bias = False)
        self.temp_avg = nn.AdaptiveAvgPool3d((1,None,None))

    def forward(self, x):
        with autocast():
            x, clip_type = x
            if (x.shape[0]) ==1:
                x = x.repeat(2,1,1,1,1)
            if clip_type == 'd':
                x = self.temp_avg(x)
                x = x.flatten(1)
                x = self.relu(self.bn1(self.fc1(x)))
                x = nn.functional.normalize(self.bn2(self.fc2(x)), p=2, dim=1)
                return x
            elif clip_type == 's':
                gsr = self.temp_avg(x)
                gsr = gsr.flatten(1)
                gsr = self.relu(self.bn1(self.fc1(gsr)))
                gsr = nn.functional.normalize(self.bn2(self.fc2(gsr)), p=2, dim=1)
                x1, x2, x3, x4 = [nn.functional.normalize(self.bn2(self.fc2(
                                    self.relu(self.bn1(self.fc1(x[:,:,i,:,:].flatten(1))))))) for i in range(4)]
                return gsr, x1, x2, x3, x4
            else:
                return None, None, None, None, None
