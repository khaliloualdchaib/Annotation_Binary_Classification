import torch
import torch.nn as nn
from math import floor

class CNN(nn.Module):
    def __init__(self, height, width):
        super(CNN, self).__init__()
        self.width = width
        self.height = height
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3,2,3,stride=2, padding=1),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Conv2d(2,4,3,stride=2, padding=1),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Conv2d(4,8,3,stride=2, padding=0),
            nn.ReLU(True),
        )
        self.fcinputsize = self.calculate_fc_input()[0] * self.calculate_fc_input()[1] * self.calculate_fc_input()[2]
        print(self.fcinputsize)
        self.fc_layers = nn.Sequential(
            nn.Linear(self.fcinputsize, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, 1),
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
            elif isinstance(module,nn.ReLU) or isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d):
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
    
