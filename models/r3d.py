# Source: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torch.cuda.amp import autocast

__all__ = ['r3d_18', 'mc3_18', 'r2plus1d_18']

model_urls = {
    'r3d_18': 'https://download.pytorch.org/models/r3d_18-b3b3357e.pth',
    'mc3_18': 'https://download.pytorch.org/models/mc3_18-a90a0ba3.pth',
    'r2plus1d_18': 'https://download.pytorch.org/models/r2plus1d_18-91a641e6.pth',
}

class Conv3DSimple(nn.Conv3d):
    def __init__(self, in_planes, out_planes, midplanes=None, stride=1, padding=1):
        super(Conv3DSimple, self).__init__(in_channels=in_planes, out_channels=out_planes, kernel_size=(3, 3, 3), stride=stride, padding=padding, bias=False)
    @staticmethod
    def get_downsample_stride(stride): return (stride, stride, stride)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(conv_builder(inplanes, planes, midplanes, stride), nn.BatchNorm3d(planes), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(conv_builder(planes, planes, midplanes), nn.BatchNorm3d(planes))
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None: residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class VideoResNet(nn.Module):
    def __init__(self, block, conv_makers, layers, stem, num_classes=400, zero_init_residual=False):
        super(VideoResNet, self).__init__()
        self.inplanes = 64
        self.stem = stem()
        self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self._initialize_weights()
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck): nn.init.constant_(m.bn3.weight, 0)
    def forward(self, x):
        with autocast():
            x, clip_type = x
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            return x, clip_type
    def _make_layer(self, block, conv_builder, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=ds_stride, bias=False), nn.BatchNorm3d(planes * block.expansion))
        layers = [block(self.inplanes, planes, conv_builder, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks): layers.append(block(self.inplanes, planes, conv_builder))
        return nn.Sequential(*layers)
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.BatchNorm3d): nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)

def _video_resnet(arch, pretrained=False, progress=True, **kwargs):
    model = VideoResNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

def r3d_18(pretrained=False, progress=True, **kwargs):
    return _video_resnet('r3d_18', pretrained, progress, block=BasicBlock, conv_makers=[Conv3DSimple] * 4, layers=[2, 2, 2, 2], stem=BasicStem, **kwargs)
