from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json
from math import floor

class FAAMDataset(Dataset):

    def __init__(self, json_file, transform=None) -> None:
        super().__init__()
        with open(json_file, "r") as f:
            self.json = json.load(f)

        self.transform = transform

    def __len__(self):
        return len(self.json["images"])
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx): #?
            idx = idx.tolist()

        img_name = os.getcwd() + self.json["images"][idx]["path"]
        image = io.imread(img_name)

        if self.transform:
            image = self.transform(image)
        return image
    
    def getAverageSize(self):
        width = 0
        height = 0
        for i in self.json["images"]:
            width += i["width"]
            height += i["height"]
        return (floor(width/len(self.json["images"])), floor(height/len(self.json["images"])))