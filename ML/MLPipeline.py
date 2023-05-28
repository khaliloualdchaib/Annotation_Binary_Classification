import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

class MLPipeline:
    def __init__(self, network, device, loss, optimizer) -> None:
        self.network = network
        self.device = device
        self.loss = loss
        self.optimizer = optimizer
    ### Training function
    def train(self, training_loader):
        ##  creating list to hold loss per batch
        loss_per_batch = []
        for images, labels in tqdm(training_loader):
            #  sending images to device
            images, labels = images.to(torch.float32).to(self.device), labels.to(self.device)
            #  classifying instances
            images = images.permute(0, 3, 1, 2)
            classifications = self.network(images)

            #  computing loss/how wrong our classifications are
            loss = self.loss(classifications, images)
            loss_per_batch.append(loss.item())

            #  zeroing optimizer gradients
            self.optimizer.zero_grad()

            #  computing gradients/the direction that fits our objective
            loss.backward()

            #  optimizing weights/slightly adjusting parameters
            self.optimizer.step()
            #print('\t partial train loss (single batch): %f' % (loss.data))

        print('Done!')

        return loss_per_batch


    def validate(self, validation_loader):
        """
        This function validates convnet parameter optimizations
        """
        #  creating a list to hold loss per batch
        loss_per_batch = []

        #  preventing gradient calculations since we will not be optimizing
        with torch.no_grad():
        #  iterating through batches
            for images, labels in tqdm(validation_loader):
                #--------------------------------------
                #  sending images and labels to device
                #--------------------------------------
                images, labels = images.to(torch.float32).to(self.device), labels.to(self.device)
                
                images = images.permute(0, 3, 1, 2)

                #--------------------------
                #  making classsifications
                #--------------------------
                classifications = self.network(images)

                #-----------------
                #  computing loss
                #-----------------
                loss = self.loss(classifications, images)
                loss_per_batch.append(loss.item())
                #print('\t partial train loss (single batch): %f' % (loss.data))

        print('Done!')
        return loss_per_batch

    def accuracy(self, dataloader):
        """
        This function computes accuracy
        """
        #  setting model state
        self.network.eval()

        network_accuracy = 0

        #  iterating through batches
        with torch.no_grad():
            for images, labels in tqdm(dataloader):
                images, labels = images.to(torch.float32).to(self.device), labels.to(self.device)

                # Flatten the input and output tensors
                images = images.permute(0, 3, 1, 2)
                outputs_flat = (self.network(images)).reshape(-1)
                inputs_flat = images.reshape(-1)    

                # Calculate the mean squared error between the input and output tensors
                mse = torch.mean(torch.square(outputs_flat - inputs_flat)) # Question input - output
                #print("Mse",mse)
                # Calculate the accuracy as the percentage of pixels that are accurately reconstructed
                accuracy = 100 * (1 - mse) # Question / torch.mean(torch.square(inputs_flat)))
                #print("accuracy",accuracy)
                network_accuracy += accuracy
        network_accuracy /= len(dataloader)
        numpy_network_accuracy = network_accuracy.cpu().numpy()

        return numpy_network_accuracy

    def test_model(self, dataloader, model):
        network_accuracy = 0
        with torch.no_grad():
            for images, labels in tqdm(dataloader):
                images, labels = images.to(torch.float32).to(self.device), labels.to(self.device)
                images = images.permute(0, 3, 1, 2)
                output = model(images)
                outputs_flat = (output).reshape(-1)
                inputs_flat = images.reshape(-1)
                # Calculate the mean squared error between the input and output tensors
                mse = torch.mean(torch.square(outputs_flat - inputs_flat)) # Question input - output
                #print("Mse",mse)
                # Calculate the accuracy as the percentage of pixels that are accurately reconstructed
                accuracy = 100 * (1 - mse) # Question / torch.mean(torch.square(inputs_flat)))
                #print("accuracy",accuracy)
                network_accuracy += accuracy
        network_accuracy /= len(dataloader)
        numpy_network_accuracy = network_accuracy.cpu().numpy()
        print("Accuracy on the test data is ", numpy_network_accuracy)
        return numpy_network_accuracy
    def train_epochs(self, epochs, training_set, validation_set, saveModel=False):

        #  creating log
        log_dict = {
            'training_loss_per_epoch': [],
            'validation_loss_per_epoch': [],
            'training_accuracy_per_epoch': [],
            'validation_accuracy_per_epoch': []
        } 

        self.network.train()

        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')
            train_losses = []

            # training
            print('Training...')
            loss_per_batch_train = self.train(training_set)
            for i in loss_per_batch_train:
                train_losses.append(i)

            print('Deriving training accuracy...')
            #  computing training accuracy
            train_accuracy = self.accuracy(training_set)
            log_dict['training_accuracy_per_epoch'].append(train_accuracy)

            #  validation
            print('Validating...')
            val_losses = []

            #  setting convnet to evaluation mode
            self.network.eval()

            loss_per_batch_validation = self.validate(validation_set)
            
            for i in loss_per_batch_validation:
                val_losses.append(i)

            print('deriving validation accuracy...')
            val_accuracy = self.accuracy(validation_set)
            log_dict['validation_accuracy_per_epoch'].append(val_accuracy)
 
            train_losses = np.array(train_losses).mean()
            log_dict['training_loss_per_epoch'].append(train_losses)
            val_losses = np.array(val_losses).mean()
            log_dict['validation_loss_per_epoch'].append(val_losses)
            print(f'training_loss: {round(train_losses, 4)}  training_accuracy: '+
            f'{train_accuracy}  validation_loss: {round(val_losses, 4)} '+  
            f'validation_accuracy: {val_accuracy}\n')
        if saveModel:
            PATH = './autoencoder.pth'
            torch.save(self.network.state_dict(), PATH)
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        # plot the training and validation loss
        ax[0].plot(log_dict['training_loss_per_epoch'], label='Training Loss')
        ax[0].plot(log_dict['validation_loss_per_epoch'], label='Validation Loss')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].legend()

        # plot the training and validation accuracy
        ax[1].plot(log_dict['training_accuracy_per_epoch'], label='Training Accuracy')
        ax[1].plot(log_dict['validation_accuracy_per_epoch'], label='Validation Accuracy')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Accuracy')
        ax[1].legend()
        # show the plot
        plt.show()
        return log_dict
    