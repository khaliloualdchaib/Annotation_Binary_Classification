from custom_datasets.FAAMDataset import *
from patches import *
from custom_datasets.PatchDataset import *
from dataTransformation.toTensor import ToTensor
from ML.MLPipeline import *
from custom_datasets.PatchDataset import *
from torch.utils.data import DataLoader
from ML.MLPipeline import * 
from datasplit import Datasplitting
from ML.CNN import CNN
import sys


#-------------------------------------------------- DATA PREP -------------------------------------------------------------------------
#Put here the path to root folder
root = os.getcwd()
dataset = FAAMDataset("jsonfiles/new.json")
patch_width = 100
patch_height = 100
def dataPrep():
    p = Patch(dataset, patch_height, patch_width, root)
    p.CreatePatches()
    Datasplitting(0.60,0.20,0.20,dataset, patch_height=patch_height, patch_width=patch_width)
print("Data prep finished")
#-------------------------------------------------- Model Training -------------------------------------------------------------------------
##################### DATA LOADING #####################################
trainingdata = PatchDataset("PatchData/Patches.csv", "PatchData/DataSplit.json", "Training")
testdata = PatchDataset("PatchData/Patches.csv", "PatchData/DataSplit.json", "Testing")
validationdata = PatchDataset("PatchData/Patches.csv", "PatchData/DataSplit.json", "Validation")
batch_size = 64
print(len(trainingdata))
print(len(testdata))
print(len(validationdata))
training_loader = DataLoader(trainingdata, batch_size=batch_size)
testing_loader = DataLoader(testdata, batch_size=batch_size)
validation_loader = DataLoader(validationdata, batch_size=batch_size)
##################### PARAMETERS #####################################
# Check if the GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = CNN(100, 100)
model.to(device)

loss_fn = torch.nn.BCELoss()
lr= 0.001

### Set the random seed for reproducible results
torch.manual_seed(0)

optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
##################### TRAINING + Validation #####################################
pipeline = MLPipeline(model, device, loss_fn, optim)
log_dict = pipeline.train_epochs(15,training_loader, validation_loader,True)
##################### TESTING #####################################
#model.load_state_dict(torch.load("autoencoder_reduced_filters.pth", map_location=torch.device('cpu')))
#model.eval()

#pipeline.test_model(testing_loader, model)