import torch
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import json

class PatchDataset(Dataset):
    def __init__(self, csv_file, DataSplitJSON, type, transform=None) -> None:
        super().__init__()
        self.transform = transform
        self.df = pd.read_csv(csv_file)
        self.type = type
        with open(DataSplitJSON, "r") as f:
            self.json = json.load(f)
    def __len__(self):
        return len(self.json[self.type])
    
    def __getitem__(self, idx):
        patchID = int(self.json[self.type][idx])
        filtered_df = self.df[(self.df["PatchID"] == patchID)]
        if filtered_df.empty:
            return "EMPTY"    
        path = self.type + "/" + str(patchID) + ".pt"
        img = torch.load(path)
        if self.transform:
            img = self.transform(img)
        #img.transpose(1,2,0) # Happens in self.transform
        #img = img[:, :, :3]
        #img = torch.from_numpy(img) # Happens in self.transform
        ############################
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #img = img.to(device)  # Move the tensor to the GPU or CPU
        ############################
        img = img.permute(1, 2, 0)
        return img, filtered_df["Annotation"].iloc[0]
    def ShowImage(self, Tensor_Image):
        # Convert the tensor to a numpy array
        image_array = Tensor_Image.numpy()
        plt.imshow(image_array)
        plt.show()
