import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.autograd import Function

import matplotlib.pyplot as plt

import random
from skimage.transform import resize



class CNN(nn.Module):
    def __init__(self, num_classes=10, num_feature=32):
        super(CNN, self).__init__()
        self.num_feature = num_feature
        self.num_classes = num_classes
        self.layer = nn.Sequential(
            nn.Conv2d(1, self.num_feature, 3, 1, 1),
            nn.BatchNorm2d(self.num_feature),
            nn.ReLU(),
            nn.Conv2d(self.num_feature, self.num_feature * 2, 3, 1, 1),
            nn.BatchNorm2d(self.num_feature * 2),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(self.num_feature * 2, self.num_feature * 4, 3, 1, 1),
            nn.BatchNorm2d(self.num_feature * 4),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(self.num_feature * 4, self.num_feature * 8, 3, 1, 1),
            nn.BatchNorm2d(self.num_feature * 8),
            nn.ReLU(),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(self.num_feature * 8 * 7 * 7, 1000),
            nn.ReLU(),
            nn.Linear(1000, self.num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaming Initialization
                init.kaiming_normal(m.weight.data)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                # Kaming Initialization
                init.kaiming_normal(m.weight.data)
                m.bias.data.fill_(0)

    def forward(self, x):
        out = self.layer(x)
        out = out.view(x.size()[0], -1)
        out = self.fc_layer(out)

        return out




if __name__ == "__main__":
    num_classes = 10
    model = CNN(num_classes).cuda()

