import torch
import torch.nn as nn
from math import floor
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, height, width):
        super(CNN, self).__init__()
        self.width = width
        self.height = height

         # onvolutional layers (3,16,32)
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size=(5, 5), stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size=(5, 5), stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=(3, 3), padding=1)

        # conected layers
        self.fc1 = nn.Linear(in_features= 64 * 6 * 6, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=1)

        #self.fcinputsize = self.calculate_fc_input()[0] * self.calculate_fc_input()[1] * self.calculate_fc_input()[2]
        

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
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)

        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x
    
