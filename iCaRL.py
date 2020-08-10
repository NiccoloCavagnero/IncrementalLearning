import numpy as np
import time
from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss

from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

from IncrementalLearning import utils

class iCaRL():
    def __init__(self,memory=2000,device='cuda',params=None,plot=False):
        self.memory = memory
        self.device = device
        self.params = params
        self.plot = plot
        
    def __SKLClassifier__(self,data,exemplars,net,n_classes,classifier):
      s = str(type(classifier)).split('.')[-1][:-2]
      print(f'\n ### {s} ###')
      net.eval()
      
      X, y = [], []
      print('   # Extract features')
      for key in range(int(n_classes/10)):
        items = self.__formatExemplars__(exemplars)

        loader = DataLoader(items, batch_size=512, shuffle=False, num_workers=4, drop_last=False)
        for images, labels in loader:
          with torch.no_grad():
            images = images.to(self.device)
            outputs = net(images,features=True)
            for output,label in zip(outputs,labels):
              X.append(np.array(output.cpu()))
              y.append(np.array(label))
    
      print(f'   # {s} Fitting ')
      classifier.fit(X,y)

      loader = DataLoader(data, batch_size=512, shuffle=False, num_workers=4, drop_last=False)

      predictions, label_list = [], []
      print(f'   # {s} Predicting ')
      for images, labels in loader:
        images = images.to(self.device)
        label_list += labels
        with torch.no_grad():
          outputs = net(images,features=True)
          for output in outputs:
            prediction = classifier.predict([np.array(output.cpu())])
            predictions.append(prediction)
          
      accuracy = accuracy_score(label_list,predictions)
      print(f'   # {s} Accuracy: {accuracy}')

      return accuracy, predictions, label_list
 
    def __updateRepresentation__(self,data,exemplars,net,n_classes,fineTune=False):
        print('\n ### Update Representation ###')
        EPOCHS = self.params['EPOCHS']
        BATCH_SIZE = self.params['BATCH_SIZE']
        LR = self.params['LR']
        MOMENTUM = self.params['MOMENTUM']
        WEIGHT_DECAY = self.params['WEIGHT_DECAY']

        # Define Loss
        criterion = BCEWithLogitsLoss()

        if len(exemplars) != 0:
          data = data + utils.formatExemplars(exemplars)
        
        # Define Dataloader
        loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

        if n_classes != 10:
          # Save network for distillation
          old_net = deepcopy(net)
          old_net.eval()
          # Update network's last layer
          net = utils.updateNet(net,n_classes)
        
        net = net.to(self.device)
        optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        
        for epoch in range(EPOCHS):
         
          # LR step down policy
          if epoch == 48 or epoch == 62:
            for g in optimizer.param_groups:
              g['lr'] = g['lr']/5
   
          # Set module in training mode
          net.train() 

          running_loss = 0.0
          for images, labels in loader:
            images = images.to(self.device)
            images = torch.stack([ utils.augmentation(image) for image in images ])
            
            # Zero-ing the gradients
            optimizer.zero_grad()
            # Forward pass to the network
            outputs = net(images)      
            # Get One Hot Encoding for the labels
            labels = utils.getOneHot(labels,n_classes)
            labels = labels.to(self.device)

            # Compute Losses
            if n_classes == 10 or fineTune:
                tot_loss = criterion(outputs, labels)
            else:
                with torch.no_grad():
                  old_outputs = torch.sigmoid(old_net(images))
                targets = torch.cat((old_outputs,labels[:,n_classes-10:]),1)
                tot_loss = criterion(outputs,targets)   

            # Update Running Loss         
            running_loss += tot_loss.item() * images.size(0)

            tot_loss.backward() 
            optimizer.step() 

          # Train loss of current epoch
          train_loss = running_loss / len(data)
          print('\r   # Epoch: {}/{}, LR = {},  Train loss = {}'.format(epoch+1, EPOCHS, optimizer.param_groups[0]['lr'], round(train_loss,5)),end='')
        print()

        return net
 
    # herding
    def __constructExemplarSet__(self,data,n_classes,net):
        print('\n ### Construct Exemplar Set ###')
        m = int(self.memory/n_classes)
        print(f'   # Exemplars per class: {m}')

        # Initialize lists of images and exemplars for each class
        class_map = utils.fillClassMap(data,n_classes)
        exemplars = dict.fromkeys(np.arange(n_classes-10,n_classes))
        for label in exemplars:
          exemplars[label] = []
        
        # Get and save net outputs for each class
        net.eval()
        for label in class_map:
          print(f'\r   # Class: {label+1}',end='')
          class_outputs = []
          mean = 0
          
          # Compute class means
          with torch.no_grad():
            loader = DataLoader(class_map[label], batch_size=512, shuffle=False, num_workers=4, drop_last=False)
            for images, _ in loader:
                images = images.to(self.device)
                outputs = net(images,features=True)
                for output in outputs:
                    class_outputs.append(output)
                    mean += output
            mean /= len(class_map[label])
          
            w_t = mean
            for i in range(m):
              maximum = -99999
              ind_max = None
              for idx,tensor in enumerate(class_outputs):
                dot = w_t.dot(tensor)

                if dot > maximum:
                  maximum = dot
                  ind_max = idx

              w_t = w_t+mean-class_outputs[ind_max]    
              class_outputs.pop(ind_max)
       
              exemplars[label].append(class_map[label][ind_max])
              class_map[label].pop(ind_max)
        print()

        return exemplars
    
    # Run ICaRL
    def run(self,train_batches,test_batches,net,herding=True,classifier='NME',NME_mode='NME'):
      t0 = time.time()
      exemplars = []
      accuracy_per_batch = []
      for idx, batch in enumerate(train_batches):
        print(f'\n##### BATCH {idx+1} #####')
        n_classes = (idx+1)*10

        # Update Representation
        net = self.__updateRepresentation__(batch,exemplars,net,n_classes)
        utils.printTime(t0)
        
        # Exemplars managing
        if herding:
          new_exemplars = self.__constructExemplarSet__(batch,n_classes,net)
        else:
          new_exemplars = utils.randomExemplarSet(self.memory,batch,n_classes)
        exemplars.update(new_exemplars)
        utils.printTime(t0)
        
        # Classification
        if classifier == 'NME':
          accuracy, predictions, labels = utils.NMEClassifier(test_batches[idx],batch,exemplars,net,n_classes,self.device)
        elif classifier == 'FC':
          accuracy, predictions, labels = utils.FCClassifier(test_batches[idx],net,n_classes,self.device)
        else:
          accuracy, predictions, labels = self.__SKLClassifier__(test_batches[idx],exemplars,net,n_classes,classifier)
        accuracy_per_batch.append(accuracy)
        utils.printTime(t0)
        
        if self.plot:
          utils.confusionMatrix(labels,predictions,idx)

        # Exemplars managing
        exemplars = utils.reduceExemplarSet(self.memory,exemplars,n_classes)
        utils.printTime(t0)

      return accuracy_per_batch
    
    # Run LwF
    def runLwF(self,train_batches,test_batches,net,fineTune=False):
      t0 = time.time()
      accuracy_per_batch = []
      for idx, batch in enumerate(train_batches):
        print(f'\n##### BATCH {idx+1} #####')
        n_classes = (idx+1)*10
        net = self.__updateRepresentation__(batch,{},net,n_classes,fineTune)
        utils.printTime(t0)
        
        accuracy, predictions, labels = utils.FCClassifier(test_batches[idx],net,n_classes,self.device)
        accuracy_per_batch.append(accuracy)
        utils.printTime(t0)
        
        if self.plot:
            utils.confusionMatrix(labels,predictions,idx)

      return accuracy_per_batch
