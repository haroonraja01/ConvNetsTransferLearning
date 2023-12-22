import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory

cudnn.benchmark = True


class FirstModel(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, 7, stride=4)
        self.conv2 = nn.Conv2d(3, 3, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(432, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x




    # def forward(self):
    #     nn.Conv2d()
