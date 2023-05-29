import torch
import torch.nn as nn
from math import floor

class CNN(nn.Module):
    def __init__(self, height, width):
        super(CNN, self).__init__()
        self.width = width
        self.height = height
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fcinputsize = self.calculate_fc_input()[0] * self.calculate_fc_input()[1] * self.calculate_fc_input()[2]

        self.fc_layers = nn.Sequential(
            nn.Linear(self.fcinputsize, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def calculate_fc_input(self):
        width = self.width
        height = self.height
        lastfilter_size = 0
        for module in self.conv_layers.children():
            if isinstance(module,nn.Conv2d):
                height = floor((height + 2* module.padding[0] - module.kernel_size[0]) / module.stride[0]) + 1
                width = floor((width + 2* module.padding[1] - module.kernel_size[1]) / module.stride[1]) + 1
            elif isinstance(module,nn.MaxPool2d):
                height = floor((height - module.kernel_size) / module.stride) + 1
                width = floor((width - module.kernel_size) / module.stride) + 1
                continue
            elif isinstance(module,nn.ReLU) or isinstance(module, nn.Dropout2d):
                continue
            elif isinstance(module,nn.BatchNorm2d):
                continue
            lastfilter_size = module.out_channels
        return lastfilter_size, height, width

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc_layers(x)
        return x
    
