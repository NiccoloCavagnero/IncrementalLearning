import numpy as np
import time
import random
from copy import deepcopy
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

from IncrementalLearning import utils

class TPCP():
    def __init__(self,memory=2000,device='cuda',params=None,plot=False):
        self.memory = memory
        self.device = device
        self.params = params
        self.plot = plot
        self.nets = []
        self.discriminator = None

    def __FCClassifier__(self,data,net,discrimination=False):
      print(f'\n ### FC Layer ###')
      print('   # FC Layer Predicting ')
      net.eval()
      
      running_corrects = 0.0
      label_list, predictions = [], []
      with torch.no_grad():
        loader = DataLoader(data, batch_size=512, shuffle=False, num_workers=4, drop_last=False)
        for images, labels in loader:
          images = images.to(self.device)
          
          if discrimination:
            labels = torch.tensor([ int(label/10) for label in labels ])
          labels = labels.to(self.device)

          outputs = torch.sigmoid(net(images))
          # Get predictions
          _, preds = torch.max(outputs.data, 1)
          # Update Corrects
          running_corrects += torch.sum(preds == labels.data).data.item()
          
          for prediction,label in zip(preds,labels):
            predictions.append(np.array(prediction.cpu()))
            label_list.append(np.array(label.cpu()))

        # Calculate Accuracy
        accuracy = running_corrects / len(data)
      
      print(f'   # FC Layer Accuracy: {accuracy}')
      return accuracy, predictions, label_list

    def __trainTask__(self,data,net):
        BATCH_SIZE = self.params['BATCH_SIZE']
        MOMENTUM = self.params['MOMENTUM']
        WEIGHT_DECAY = self.params['WEIGHT_DECAY']
        EPOCHS = self.params['EPOCHS']
        LR = self.params['LR']

        # Define Loss
        criterion = MSELoss() 
        # Define Dataloader
        loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

        net.fc = nn.Linear(64,10)
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
            labels = utils.getOneHot(labels,10)
            labels = labels.to(self.device)

            # Zero-ing the gradients
            optimizer.zero_grad()
            # Forward pass to the network
            outputs = torch.sigmoid(net(images))

            # Compute Losses
            tot_loss = criterion(outputs,labels)
                
            # Update Running Loss         
            running_loss += tot_loss.item() * images.size(0)

            tot_loss.backward() 
            optimizer.step() 

          # Train loss of current epoch
          train_loss = running_loss / len(data)
          print('\r   # Epoch: {}/{}, LR = {},  Train loss = {}'.format(epoch+1, EPOCHS, optimizer.param_groups[0]['lr'], round(train_loss,5)),end='')
        print()

        self.nets.append(deepcopy(net))

        return net

    def __trainDiscriminator__(self,net,exemplars,n_tasks):
        BATCH_SIZE = self.params['BATCH_SIZE']
        MOMENTUM = self.params['MOMENTUM']
        WEIGHT_DECAY = self.params['WEIGHT_DECAY']
        EPOCHS = self.params['EPOCHS']
        LR = self.params['LR']

        data = self.__formatExemplars__(exemplars)

        # Define Loss
        criterion = MSELoss() 
        # Define Dataloader
        loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

        net.fc = nn.Linear(64,n_tasks)

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
            labels = torch.tensor([ int(label/10) for label in labels ])
            labels = utils.getOneHot(labels,n_classes)
            labels = labels.to(self.device)

            # Zero-ing the gradients
            optimizer.zero_grad()
            # Forward pass to the network
            outputs = torch.sigmoid(net(images))

            # Compute Losses
            tot_loss = criterion(outputs,labels)
                
            # Update Running Loss         
            running_loss += tot_loss.item() * images.size(0)

            tot_loss.backward() 
            optimizer.step() 

          # Train loss of current epoch
          train_loss = running_loss / len(data)
          print('\r   # Epoch: {}/{}, LR = {},  Train loss = {}'.format(epoch+1, EPOCHS, optimizer.param_groups[0]['lr'], round(train_loss,5)),end='')
        print()

        discriminator = deepcopy(net)

        return net

    def __randomExemplarSet__(self,data,n_classes):
      print('\n ### Construct Random Exemplar Set ###')
      if n_classes != 10:
        m = int(self.memory/(n_classes-10))
      else:
        m = int(self.memory/(n_classes))
      print(f'   # Exemplars per class: {m}')

      # Initialize lists of images and exemplars for each class
      class_map = utils.fillClassMap(data,n_classes)
      exemplars = dict.fromkeys(np.arange(n_classes-10,n_classes))
      for label in exemplars:
        exemplars[label] = []

      for label in class_map:
        indexes = random.sample(range(len(class_map[label])),m)   
        for idx in indexes:
            exemplars[label].append(class_map[label][idx])

      return exemplars
  
    def __reduceExemplarSet__(self,exemplars,n_classes):
      print('\n ### Reduce Exemplar Set ###')
      m = int(self.memory/n_classes)
      print(f'   # Exemplars per class: {m}')
      for key in exemplars:
        exemplars[key] = exemplars[key][:m]
      
      return exemplars
    
    # dict to list
    def __formatExemplars__(self,exemplars):
      new_exemplars = []
      for key in exemplars:
        for item in exemplars[key]:
          new_exemplars.append([item[0],item[1]])

      return new_exemplars

    def __run__(self,train_batches,test_batches,net):
      t0 = time.time()
      exemplars = {}
      accuracy_per_batch = []
      for idx, batch in enumerate(train_batches):
        print(f'\n##### BATCH {idx+1} #####')
        n_classes = (idx+1)*10

        # Update Representation
        net = self.__trainTask__(batch,net)
        utils.printTime(t0)
        
        new_exemplars = self.__randomExemplarSet__(batch,n_classes)
        exemplars.update(new_exemplars)
        utils.printTime(t0)
        
        if idx == 1:
          self.discriminator = self.__trainDiscriminator__(net,exemplars,idx+1)
        elif idx > 1:
          self.discriminator = self.__trainDiscriminator__(self.discriminator,exemplars,idx+1)

        # Classifier
        if idx != 0:
            self.__FCClassifier__(test_batches[idx],self.discriminator,True)
        
        # Exemplars managing
        exemplars = self.__reduceExemplarSet__(exemplars,n_classes)
        utils.printTime(t0)

      return accuracy_per_batch

