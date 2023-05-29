import csv # for def Datasplitting
import math # for def Datasplitting
import random # for def Datasplitting
import json
import os
from tqdm import tqdm

def Datasplitting(trainingpercentage, testingpercentage, validationpercentage, dataset, patch_height, patch_width, cleansize = 8504, annotatedsize = 2962):
    totalsize = cleansize + annotatedsize # totalsize of patches

    # Create lists this should be made in patches.py for less computation

    clean = {}
    annotated = {}

    with open('PatchData/patches.csv', 'r') as file: # need to get as parameter
        reader = csv.DictReader(file)
        for row in reader:
            if row["Annotation"] == "True":
                if(row["OriginalImageID"] in annotated):
                    annotated[row["OriginalImageID"]].append(row["PatchID"])
                else:
                    annotated[row["OriginalImageID"]] = [row["PatchID"]]
            else:
                if(row["OriginalImageID"] in clean):
                    clean[row["OriginalImageID"]].append(row["PatchID"])
                else:
                    clean[row["OriginalImageID"]] = [row["PatchID"]]

    #print("annotated",annotated)


    

    Training_dataset = {}
    Testing_dataset = {}
    Validation_dataset = {}
    #### CREATE TRAINING DATASET
    for i in tqdm(clean, desc="CREATE TRAINING DATASET"):    
        #amount_patches_vertical = dataset[int(i)].shape[1] / patch_height
        #amount_patches_horizontal = dataset[int(i)].shape[0] / patch_width
        #patch_amount = math.ceil(amount_patches_vertical) * math.ceil(amount_patches_horizontal)
        amount_per_image = math.floor(int(len(annotated[i]) * trainingpercentage))
        amount_per_image_testing = math.floor(int(len(annotated[i]) * testingpercentage))
        amount_per_image_validation = math.floor(int(len(annotated[i]) * validationpercentage))
        random.seed(42)
        random.shuffle(clean[i])
        random.shuffle(annotated[i])
        Training_dataset[i] = clean[i][:amount_per_image]
        Training_dataset[i].extend(annotated[i][:amount_per_image])

        clean[i] = clean[i][amount_per_image:]
        annotated[i] = annotated[i][amount_per_image:]

        Validation_dataset[i] = clean[i][:amount_per_image_validation]
        Validation_dataset[i].extend(annotated[i][:amount_per_image_validation])
        clean[i] = clean[i][amount_per_image_validation:]
        annotated[i] = annotated[i][amount_per_image_validation:]


        Testing_dataset[i] = clean[i][:amount_per_image_testing]
        Testing_dataset[i].extend(annotated[i][:amount_per_image_testing])
        clean[i] = clean[i][amount_per_image_testing:]
        annotated[i] = annotated[i][amount_per_image_testing:]
    


    training_path = os.getcwd() + "\\Training\\"
    testing_path = os.getcwd() + "\\Testing\\"
    validation_path = os.getcwd() + "\\Validation\\"
    Patches_path = os.getcwd() + "\\Patches\\"
    if not os.path.exists(training_path):
        os.makedirs(training_path)
    if not os.path.exists(testing_path):
        os.makedirs(testing_path)
    if not os.path.exists(validation_path):
        os.makedirs(validation_path)
    for i in tqdm(Training_dataset, desc="Moving training patches"):
        for j in Training_dataset[i]:
            source = Patches_path + i + "\\" + j + ".pt"
            destination = training_path + j + ".pt"
            os.rename(source,destination)
    for i in tqdm(Testing_dataset, desc="Moving testing patches"):
        for j in Testing_dataset[i]:
            source = Patches_path + i + "\\" + j + ".pt"
            destination = testing_path + j + ".pt"
            os.rename(source,destination)
    for i in tqdm(Validation_dataset, desc="Moving validation patches"):
        for j in Validation_dataset[i]:
            source = Patches_path + i + "\\" + j + ".pt"
            destination = validation_path + j + ".pt"
            os.rename(source,destination)
    #get list of all patches for each set
    TrainingList = [patchid for list in Training_dataset.values() for patchid in list]
    TestingList = [patchid for list in Testing_dataset.values() for patchid in list]
    ValidationList = [patchid for list in Validation_dataset.values() for patchid in list]
    file = {"Training": TrainingList, "Testing": TestingList, "Validation": ValidationList}
    #Making the json file
    with open("PatchData/DataSplit.json", "w") as f:
        json.dump(file, f)