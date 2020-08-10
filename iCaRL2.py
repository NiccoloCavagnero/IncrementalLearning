import numpy as np
import time
from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import MSELoss

from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

from IncrementalLearning import utils

class iCaRL2():
    def __init__(self,memory=2000,device='cuda',params=None,plot=False):
        self.memory = memory
        self.device = device
        self.params = params
        self.plot = plot
        self.teachers = []
 
    def __train__(self,data,exemplars,net,n_classes,stabilize=False):
        step = int(n_classes/10) - 1
        BATCH_SIZE = self.params['BATCH_SIZE']
        MOMENTUM = self.params['MOMENTUM']
        WEIGHT_DECAY = self.params['WEIGHT_DECAY']
        lambda_ = self.params['lambda']

        if not stabilize:
          print('\n ### Update Representation ###')
          WEIGHT_DECAY = np.linspace(WEIGHT_DECAY,WEIGHT_DECAY/10,10)[step]
          EPOCHS = self.params['EPOCHS']
          LR = self.params['LR']
          delta = self.params['delta']
          lambda_ += delta * ( step - 1 )
          milestones = set([ 49, 63 ])
          
          if len(exemplars) != 0:
            data = data + utils.formatExemplars(exemplars)
            # Save network for distillation
            old_net = deepcopy(net)
            old_net.eval()
            self.teachers.append(old_net)
            # Update network's last layer
            net = utils.updateNet(net,n_classes)
        
        else:
          print('\n ### Stabilize Network ###')
          EPOCHS = self.params['EPOCHS2']
          LR = self.params['LR2']
          milestones = set([ int(EPOCHS/3), int(2*EPOCHS/3) ])
          data = utils.formatExemplars(exemplars)

        # Define Loss
        criterion = MSELoss() 
        # Define Dataloader
        loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
        net = net.to(self.device)
        optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        
        for epoch in range(EPOCHS):
         
          # LR step down policy
          if epoch+1 in milestones:
            for g in optimizer.param_groups:
              g['lr'] = g['lr']/5
   
          # Set module in training mode
          net.train() 

          running_loss = 0.0
          for images, labels in loader:
            # Data augmentation
            images = images.to(self.device)
            images = torch.stack([ utils.augmentation(image) for image in images ])
            # Get One Hot Encoding for the labels
            labels = utils.getOneHot(labels,n_classes)
            labels = labels.to(self.device)

            # Zero-ing the gradients
            optimizer.zero_grad()
            # Forward pass to the network
            outputs = torch.sigmoid(net(images))

            # Compute Losses
            if n_classes == 10 or stabilize:
                tot_loss = criterion(outputs,labels)
            else:
                with torch.no_grad():
                  old_outputs = torch.sigmoid(self.__getOldOutputs__(n_classes,images))
                class_loss = criterion(outputs,labels)
                distill_loss = criterion(torch.pow(outputs[:,:n_classes-10],1/2),torch.pow(old_outputs,1/2))
                tot_loss = class_loss + distill_loss * lambda_
                
            # Update Running Loss         
            running_loss += tot_loss.item() * images.size(0)

            tot_loss.backward() 
            optimizer.step() 

          # Train loss of current epoch
          train_loss = running_loss / len(data)
          print('\r   # Epoch: {}/{}, LR = {},  Train loss = {}'.format(epoch+1, EPOCHS, optimizer.param_groups[0]['lr'], round(train_loss,5)),end='')
        print()

        return net
    
    def __getOldOutputs__(self,n_classes,images):
      with torch.no_grad():
        for i in range(int(n_classes / 10)-1):
          if i == 0:
            outputs = self.teachers[i](images)
          else:
            current_outputs = self.teachers[i](images)[:,i*10:i*10+10]
            outputs = torch.cat((outputs,current_outputs),1)
      return outputs
      
    # Run ICaRL
    def run(self,train_batches,test_batches,net):
      t0 = time.time()
      exemplars = {}
      accuracy_per_batch = []
      for idx, batch in enumerate(train_batches):
        print(f'\n##### BATCH {idx+1} #####')
        n_classes = (idx+1)*10

        # Update Representation
        net = self.__train__(batch,exemplars,net,n_classes)
        utils.printTime(t0)
        
        # Exemplars managing
        new_exemplars = utils.randomExemplarSet(self.memory,batch,n_classes)
        exemplars.update(new_exemplars)
        utils.printTime(t0)
        
        # Stabilization 
        if idx != 0:
          utils.FCClassifier(test_batches[idx],net,n_classes,self.device)
          utils.printTime(t0)
          net = self.__train__([],exemplars,net,n_classes,stabilize=True)
          utils.printTime(t0)
        
        # Classification
        utils.FCClassifier(test_batches[idx],net,n_classes,self.device)
        utils.printTime(t0)
        accuracy, predictions, labels = utils.NMEClassifier(test_batches[idx],batch,exemplars,net,n_classes,self.device)
        accuracy_per_batch.append(accuracy)
        utils.printTime(t0)
        
        if self.plot:
          utils.confusionMatrix(labels,predictions,idx)

        # Exemplars managing
        exemplars = utils.reduceExemplarSet(self.memory,exemplars,n_classes)
        utils.printTime(t0)

      return accuracy_per_batch
