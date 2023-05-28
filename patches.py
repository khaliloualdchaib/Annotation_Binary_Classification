from custom_datasets.FAAMDataset import *
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm
from torchvision.transforms import ToTensor
from PIL import Image

class Patch:
    def __init__(self, data, height, width, root) -> None:
        self.data = data
        self.imageheight, self.imagewidth = self.data[0].shape[1], self.data[0].shape[0]
        self.height = height
        self.width = width
        self.patchcounter = 0
        self.root = root
        self.csv = {
            'PatchID': [],
            'OriginalImageID': [],
            'TopleftX': [], 
            'TopleftY': [], 
            'BottomrightX': [],
            'BottomrightY': [],
            'Annotation': []
        }
        self.clean = []
        self.annotated = []

    
    def CreatePatchesImage(self, image, index):
        self.imageheight, self.imagewidth = self.data[index].shape[1], self.data[index].shape[0]
        folder_path = self.root + "\\Patches\\" + str(index)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        for w in range(0, self.imagewidth, self.width):
            for h in range(0, self.imageheight, self.height):
                if w + self.width >= self.imagewidth and h + self.height >= self.imageheight:
                    overlapheight = (h+self.height) - self.imageheight
                    overlapwidth = (w+self.width) - self.imagewidth
                    crop_right = self.imagewidth
                    crop_bottom = self.imageheight
                    crop_left = w - overlapwidth
                    crop_top = h - overlapheight
                # overlap x-as
                elif w + self.width >= self.imagewidth:
                    overlap = (w+self.width) - self.imagewidth
                    crop_right = self.imagewidth
                    crop_bottom = h + self.height
                    crop_left = w - overlap
                    crop_top = h
                    
                elif h + self.height >= self.imageheight:
                    overlap = (h+self.height) - self.imageheight
                    crop_right = w + self.width
                    crop_bottom = self.imageheight
                    crop_left = w
                    crop_top = h - overlap

                else:
                    crop_right = w + self.width
                    crop_bottom = h + self.height
                    crop_left = w
                    crop_top = h


                annotation_label = None
                annotations_image = self.data.json["images"][index]["annotations"]
                if(len(annotations_image) == 0):
                    annotation_label = False
                    return
                else:
                    for annotation in annotations_image:
                        segmentations = annotation["segmentation"][0]
                        if annotation_label == True: #annotated
                            break

                        for i in range(0,len(segmentations),2):
                            segmentation = [segmentations[i],segmentations[i+1]]
                            #print(segmentation,crop_left,crop_top,crop_right,crop_bottom)
                            if self.is_segmentation_in_patch(segmentation,crop_left,crop_top,crop_right,crop_bottom):
                                annotation_label = True #annotated                             
                                break
                            else:
                                annotation_label = False


                img = image[crop_left:crop_right, crop_top:crop_bottom]
                self.csv['PatchID'].append(self.patchcounter)
                self.csv['OriginalImageID'].append(index)
                self.csv["TopleftX"].append(crop_left)
                self.csv["TopleftY"].append(crop_top)
                self.csv["BottomrightX"].append(crop_right)
                self.csv["BottomrightY"].append(crop_bottom)
                self.csv["Annotation"].append(annotation_label)
                pil_image = Image.fromarray(img)
                totensor = ToTensor()
                tensor_image = totensor(pil_image)
                torch.save(tensor_image, folder_path+'/'+str(self.patchcounter) + '.pt')
                if(annotation_label == True):
                    self.annotated.append(self.patchcounter)
                else:
                    self.clean.append(self.patchcounter)
                self.patchcounter += 1
    def CreatePatches(self):
        for i in tqdm(range(len(self.data)), desc="CREATE PATCHES"):
            self.CreatePatchesImage(self.data[i], i)
        df = pd.DataFrame(self.csv)
        df.to_csv('PatchData/Patches.csv')

    def is_segmentation_in_patch(self, segmentation, patch_left, patch_top, patch_right, patch_bottom):
        if patch_left <= segmentation[1] <= patch_right and patch_top <= segmentation[0] <= patch_bottom:
            return True
        return False