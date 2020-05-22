import numpy as np
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss

# One hot
def get_one_hot(target,num_class):
    one_hot=torch.zeros(target.shape[0],num_class).to(device)
    one_hot=one_hot.scatter(dim=1,index=target.long().view(-1,1),value=1.)
    return one_hot

class LwF():
    def __init__(self,memory=2000,device='cuda',params=None, train_batches, test_batches):
        self.memory = memory
        self.device = device
        self.params = params
        self.train_batches = train_batches
        self.test_batches = test_batches
        
        
    def dataloaders(self, train, test):
        train_loader = DataLoader(dataset=train,
                                  shuffle=True,
                                  batch_size=self.batchsize)

        test_loader = DataLoader(dataset=test,
                                 shuffle=True,
                                 batch_size=self.batchsize)

        return train_loader, test_loader

def train(self,new_data,exemplars,net,n_classes):
        print('\n ### Updating Representation ###')
        EPOCHS = self.params['EPOCHS']
        BATCH_SIZE = self.params['BATCH_SIZE']
        LR = self.params['LR']
        MOMENTUM = self.params['MOMENTUM']
        WEIGHT_DECAY = self.params['WEIGHT_DECAY']

        # Define Loss
        criterion = BCEWithLogitsLoss()
        
        # Concatenate new data with set of exemplars
        if len(exemplars) != 0:
          data = new_data + exemplars
        else:
          data = new_data
        
        # Define Dataloader
        loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

        if n_classes != 10:
          # Store network outputs with pre-update parameters
          old_outputs = self.__getOldOutputs__(loader,net,n_classes-10)
        
          # Update network's last layer
          net = self.__updateNet__(net,n_classes)

        optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        
        net = net.to(self.device)
        
        for epoch in range(EPOCHS):

          # LR step down policy
          if epoch == 48 or epoch == 62:
            for g in optimizer.param_groups:
              g['lr'] = g['lr']/5
            
          net.train() # Sets module in training mode

          running_loss = 0.0

          for images, labels, indexes in loader:
            indexes = indexes.to(self.device)              
            images = images.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad() # Zero-ing the gradients

            # Forward pass to the network
            outputs = net(images)
                
            # Compute Losses
            labels = self.__getOneHot__(labels,n_classes)
            class_loss = criterion(outputs[:,n_classes-10:], labels[:,n_classes-10:])

            if n_classes != 10:
              distill_loss = criterion(outputs[:,:n_classes-10], old_outputs[indexes])
              tot_loss = class_loss + distill_loss
            else:
              tot_loss = class_loss              
            # Update Running Loss
            running_loss += tot_loss.item() * images.size(0)

            # Compute gradients for each layer and update weights
            tot_loss.backward() 

            optimizer.step() # update weights based on accumulated gradients
            
          # Train loss of current epoch
          train_loss = running_loss / len(data)
          print('\r   # Epoch: {}/{}, LR = {},  Train loss = {}'.format(epoch+1, EPOCHS, optimizer.param_groups[0]['lr'], round(train_loss,5)),end='')
        print()

        return net
