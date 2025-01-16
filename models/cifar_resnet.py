import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

from functools import partial

class CIFAR10ResNet(models.resnet.ResNet):
    def __init__(self, arch='resnet34', num_classes=2048, **kwargs):
        if arch == 'resnet34':
            block = BasicBlock
        elif arch == 'resnet50':
            block = Bottleneck
        else:
            raise Exception("architecture {arch} not supported")
        
        super().__init__(block, [3, 4, 6, 3], num_classes=num_classes, **kwargs)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.Identity()
        self.bn_final = nn.BatchNorm1d(self.fc.in_features, affine=False, eps=1e-6)
        
        self.arch = arch 

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward with L2 normalization applied after flattening and before fc
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.bn_final(x) # is this still needed?
        x = self.fc(x)      # this always gets redefined to Identity in MEC, have to redefine when doing validation

        return x


cifar_resnet34 = partial(CIFAR10ResNet, arch='resnet34')
cifar_resnet50 = partial(CIFAR10ResNet, arch='resnet50')
        