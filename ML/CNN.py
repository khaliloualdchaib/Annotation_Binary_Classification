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
        self.fc1 = nn.Linear(in_features=2304, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=1)

        self.dropout = nn.Dropout(p=0.5)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)

        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))

        return x
    
