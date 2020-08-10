import numpy as np
import time
import random
from copy import deepcopy
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import MSELoss, BCELoss
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

    def __FCClassifier__(self,data,net,task,discrimination=False,print_=True):
      if print_:
        print(f'\n ### FC Layer ###')
        if discrimination:
          print('   # FC Layer Discriminating ')
        else:
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
          else:
            labels = torch.tensor([ label-(task*10) for label in labels ])

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
      
      if print_:
        print(f'   # FC Layer Accuracy: {accuracy}')

      return accuracy, predictions, label_list

    def __trainTask__(self,data,net,n_classes):
        print('Training task')
        BATCH_SIZE = self.params['BATCH_SIZE']
        MOMENTUM = self.params['MOMENTUM']
        WEIGHT_DECAY = self.params['WEIGHT_DECAY']
        EPOCHS = self.params['EPOCHS']
        LR = self.params['LR']
        milestones = set([ int(7/10*EPOCHS), int(9/10*EPOCHS) ])

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
            labels = torch.tensor([ label-(n_classes-10) for label in labels ])
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

    def __trainDiscriminator__(self,net,exemplars,n_tasks,test_data):
        print('Training discriminator')
        BATCH_SIZE = self.params['BATCH_SIZE']
        MOMENTUM = self.params['MOMENTUM']
        WEIGHT_DECAY = self.params['WEIGHT_DECAY']
        EPOCHS = self.params['EPOCHS2']
        LR = self.params['LR2']
        milestones = set([ int(7/10*EPOCHS), int(9/10*EPOCHS) ])

        data = utils.formatExemplars(exemplars)

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
            labels = utils.getOneHot(labels,n_tasks)
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
          acc, _, _ = self.__FCClassifier__(test_data,net,n_tasks,True,False)

          print('\r   # Epoch: {}/{}, LR = {},  Train loss = {}, Test accuracy = {}'.format(epoch+1, EPOCHS, optimizer.param_groups[0]['lr'], round(train_loss,5), round(acc,5)),end='')
        print()

        return net

    def run(self,train_batches,test_batches,net,net2):
      t0 = time.time()
      exemplars = {}
      accuracy_per_batch = []
    
      for idx, batch in enumerate(train_batches):
        print(f'\n##### BATCH {idx+1} #####')
        n_classes = (idx+1)*10
        
        new_exemplars = utils.randomExemplarSet(self.memory,batch,n_classes)
        exemplars.update(new_exemplars)
        utils.printTime(t0)

        if idx != 0:
          self.discriminator = self.__trainDiscriminator__(self.discriminator,exemplars,idx+1,test_batches[idx])
          _, tasks, _ = self.__FCClassifier__(test_batches[idx],self.discriminator,n_classes,True)
        
        # Update Representation
        net = self.__trainTask__(batch,net,n_classes)
        utils.printTime(t0)

        if idx == 0:
            self.discriminator = self.__trainTask__(batch,net,n_classes)
        
        # Classifier
        else:
            task_dict = dict.fromkeys([i for i in range(idx+1)])
            for key in task_dict:
              task_dict[key] = []

            for item, task in zip(test_batches[idx], tasks):
              print(task)
              task_dict[int(task)].append(item)

            tot_acc = 0.0
            
            for task in task_dict:
              acc, _, _ = self.__FCClassifier__(task_dict[task],self.nets[task],task,False)
              tot_acc += acc * len(task_dict[task])
            
            tot_acc /= len(test_batches[idx])

            print(f"Total Accuracy: {tot_acc/(idx+1)}")
        
        # Exemplars managing
        exemplars = utils.reduceExemplarSet(self.memory,exemplars,n_classes)
        utils.printTime(t0)

      return accuracy_per_batch
