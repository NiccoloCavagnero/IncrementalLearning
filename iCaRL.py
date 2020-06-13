import numpy as np
import time
import random 
from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss, functional as F

import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

class iCaRL():
    def __init__(self,memory=2000,device='cuda',params=None,plot=False):
        self.memory = memory
        self.device = device
        self.params = params
        self.plot = plot

    def __NMEClassifier__(self,data,batch,exemplars,net,n_classes,mode='NME'):
      print(f'\n ### NME ###')
      means = dict.fromkeys(np.arange(n_classes))
      net.eval()

      batch_map = self.__fillClassMap__(batch,n_classes)

      print('   # Computing means ')
      for key in range(n_classes):
        if key in range(n_classes-10,n_classes):
          items = batch_map[key]
        else:
          items = exemplars[key]
        
        loader = DataLoader(items, batch_size=512, shuffle=False, num_workers=4, drop_last=False)
        mean = torch.zeros((1,64),device=self.device)
        for images, _ in loader:
          with torch.no_grad():
            images = images.to(self.device)
            flipped_images = torch.flip(images,[3])
            images = torch.cat((images,flipped_images))
            
            outputs = net(images,features=True)
            
            for output in outputs:
              mean += output
        mean = mean / ( 2 * len(items) ) 
        means[key] = mean / mean.norm()

      loader = DataLoader(data, batch_size=512, shuffle=False, num_workers=4, drop_last=False)

      n_correct = 0.0
      print('   # NME Predicting ')
      predictions = []
      label_list = []
      for images, labels in loader:
        images = images.to(self.device)
        label_list += list(labels)
        with torch.no_grad():
          outputs = net(images,features=True)
          for output in outputs:
            prediction = None
            if mode == 'NME':
              min_dist = 99999
              for key in means:
                dist = torch.dist(means[key],output)
                if dist < min_dist:
                  min_dist = dist
                  prediction = key
            elif mode == 'Cosine':
              max_similarity = 0
              for key in means:
                cosine = torch.sum(means[key]*output)
                if cosine > max_similarity:
                  max_similarity = cosine
                  prediction = key
            predictions.append(prediction)
          
      for label, prediction in zip(label_list,predictions):
        if label == prediction:
          n_correct += 1
      
      accuracy = n_correct/len(data)
      print(f'   # NME Accuracy: {accuracy}')

      return accuracy, predictions, label_list

    def __FCClassifier__(self,data,net,n_classes):
      print(f'\n ### FC Layer ###')
      print('   # FC Layer Predicting ')
      net.eval()
      
      running_corrects = 0.0
      with torch.no_grad():
        loader = DataLoader(data, batch_size=512, shuffle=False, num_workers=4, drop_last=False)
        for images, labels in loader:
          images = images.to(self.device)
          labels = labels.to(self.device)

          outputs = torch.sigmoid(net(images))
          # Get predictions
          _, preds = torch.max(outputs.data, 1)
          # Update Corrects
          running_corrects += torch.sum(preds == labels.data).data.item()

        # Calculate Accuracy
        accuracy = running_corrects / len(data)
      
      print(f'   # FC Layer Accuracy: {accuracy}')

      return accuracy

    def __SKLClassifier__(self,data,exemplars,net,n_classes,classifier):
      s = str(type(classifier)).split('.')[-1][:-2]
      print(f'\n ### {s} ###')
      net.eval()
      
      X = []
      y = []
      print('   # Extract features')
      for key in exemplars:
        loader = DataLoader(exemplars[key], batch_size=512, shuffle=False, num_workers=4, drop_last=False)
        mean = torch.zeros((1,64),device=self.device)
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

      n_correct = 0.0
      print(f'   # {s} Predicting ')
      for images, labels in loader:
        images = images.to(self.device)
        with torch.no_grad():
          outputs = net(images,features=True)
          predictions = []
          for output in outputs:
            prediction = classifier.predict([np.array(output.cpu())])
            predictions.append(prediction)
          
          for label, prediction in zip(labels,predictions):
            if label == prediction[0]:
              n_correct += 1
      
      accuracy = n_correct/len(data)
      print(f'   # {s} Accuracy: {accuracy}')

      return accuracy
 
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
          data = data + exemplars
        
        old_net = deepcopy(net)
        old_net.eval()
        
        # Define Dataloader
        loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

        if n_classes != 10:
          # Update network's last layer
          net = self.__updateNet__(net,n_classes)
          
        optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        net = net.to(self.device)
        
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
            images = torch.stack([ self.__augmentation__(image) for image in images ])
            labels = labels.to(self.device)
            
            # Zero-ing the gradients
            optimizer.zero_grad()
            # Forward pass to the network
            outputs = net(images)      
            # Get One Hot Encoding for the labels
            labels = self.__getOneHot__(labels,n_classes)

            # Compute Losses
            if n_classes == 10 or fineTune:
                tot_loss = criterion(outputs[:,n_classes-10:], labels[:,n_classes-10:])
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

    def __randomExemplarSet__(self,data,n_classes):
      print('\n ### Construct Random Exemplar Set ###')
      m = int(self.memory/n_classes)

      # Initialize list of means, images and exemplars for each class
      class_map = self.__fillClassMap__(data,n_classes)
      exemplars = dict.fromkeys(np.arange(n_classes-10,n_classes))

      for label in exemplars:
        exemplars[label] = []

      for label in range(n_classes-10,n_classes):
        indexes = random.sample(range(len(class_map[label])),m)   
        for i in indexes:
          exemplars[label].append(class_map[label][i])

      return exemplars
    
    def __constructExemplarSet__(self,data,n_classes,net):
        print('\n ### Construct Exemplar Set ###')
        m = int(self.memory/n_classes)

        # Initialize list of means, images and exemplars for each class
        class_map = self.__fillClassMap__(data,n_classes)
        means = dict.fromkeys(np.arange(n_classes-10,n_classes))
        exemplars = dict.fromkeys(np.arange(n_classes-10,n_classes))

        for label in exemplars:
          exemplars[label] = []
        
        # Get and save net outputs for each class
        net.eval()
        for label in class_map:
          print(f'\r   # Class: {label+1}',end='')
          mean = torch.zeros((1,64),device=self.device)
          class_outputs = []
          
          # Compute class means
          with torch.no_grad():
            loader = DataLoader(class_map[label], batch_size=512, shuffle=False, num_workers=4, drop_last=False)
            for images, _ in loader:
                images = images.to(self.device)
                outputs = net(images,features=True)
                for output in outputs:
                    output = output.to(self.device)
                    class_outputs.append(output)
                    mean += output
            mean = mean / len(class_map[label])
            means[label] = mean / mean.norm()
          
          # Construct exemplar list for current class
          for i in range(m):
            min_distance = 99999
            exemplar_sum = 0
            for idx, tensor in enumerate(class_outputs):
              temp_tensor = (exemplar_sum + tensor) / (len(exemplars[label])+1)
              temp_tensor = temp_tensor / temp_tensor.norm()

              # Update when a new distance is < than min_distance
              if torch.dist(mean,temp_tensor) < min_distance:
                min_distance = torch.dist(mean,temp_tensor)
                min_index = idx
                
            exemplar_sum += class_outputs[min_index]
            
            exemplars[label].append(class_map[label][min_index])
            class_map[label].pop(min_index)
            class_outputs.pop(min_index)
        print()

        return exemplars

    def __reduceExemplarSet__(self,exemplars,n_classes):
      print('\n ### Reduce Exemplar Set ###')
      m = int(self.memory/n_classes)
      print(f'   # Exemplars per class: {m}')
      for key in exemplars:
        exemplars[key] = exemplars[key][:m]
      
      return exemplars

    def __updateNet__(self,net,n_classes):
      in_features = net.fc.in_features
      out_features = net.fc.out_features
      weight = net.fc.weight.data
      bias = net.fc.bias.data

      net.fc = nn.Linear(in_features, n_classes)
      net.fc.weight.data[:out_features] = weight
      net.fc.bias.data[:out_features] = bias

      return net

    def __getOneHot__(self, target, n_classes):
      one_hot = torch.zeros(target.shape[0], n_classes).to(self.device)
      one_hot = one_hot.scatter(dim=1,index=target.long().view(-1,1),value=1.)
      
      return one_hot

    def __formatExemplars__(self,exemplars):
      new_exemplars = []
      for key in exemplars:
        for item in exemplars[key]:
          new_exemplars.append([item[0],item[1]])

      return new_exemplars

    def __augmentation__(self,image):
      image = F.pad(image,(4,4,4,4),value=-1)
      x = random.randint(0,7)
      y = random.randint(0,7)
      image = image[:,x:x+32,y:y+32]
      if random.randint(0,1):
        image = torch.flip(image,[2])
      return image

    def __confusionMatrix__(self,labels,predictions):
      cm = confusion_matrix(labels, predictions)
      for i in range(cm.shape[0]):
        for j in range(cm.shape[0]):
          cm[i][j] = np.log(1+cm[i][j])
      sns.heatmap(cm,cmap='jet',cbar=False)
      plt.show()

    def __fillClassMap__(self,data,n_classes):
      class_map = dict.fromkeys(np.arange(n_classes-10,n_classes))
      
      for label in class_map:
        class_map[label] =  []
        
      # Fill class_map
      for item in data:
        for label in class_map:
          if item[1] == label:
            class_map[label].append(item)

      return class_map
      
    def __printTime__(self,t0):
      print(f'\n   # Elapsed time: {round((time.time()-t0)/60,2)}')
    
    # Run ICaRL
    def run(self,train_batches,test_batches,net,herding=True,classifier='NME',NME_mode='NME'):
      t0 = time.time()
      exemplars = {}
      new_exemplars = []
      accuracy_per_batch = []
      for idx, batch in enumerate(train_batches):
        print(f'\n##### BATCH {idx+1} #####')
        n_classes = (idx+1)*10

        # Update Representation
        net = self.__updateRepresentation__(batch,new_exemplars,net,n_classes)
        self.__printTime__(t0)
        
        # Classifier
        if classifier == 'NME':
          accuracy, predictions, labels = self.__NMEClassifier__(test_batches[idx],batch,exemplars,net,n_classes,NME_mode)
        elif classifier == 'FC':
          accuracy = self.__FCClassifier__(test_batches[idx],net,n_classes)
        else:
          accuracy = self.__SKLClassifier__(test_batches[idx],exemplars,net,n_classes,classifier)
        accuracy_per_batch.append(accuracy)
        self.__printTime__(t0)
        
        if self.plot:
          self.__confusionMatrix__(labels,predictions)

        # Exemplars managing
        exemplars = self.__reduceExemplarSet__(exemplars,n_classes)
        self.__printTime__(t0)
        
        if herding:
          new_exemplars = self.__constructExemplarSet__(batch,n_classes,net)
        else:
          new_exemplars = self.__randomExemplarSet__(batch,n_classes)
        exemplars.update(new_exemplars)
        new_exemplars = self.__formatExemplars__(exemplars)
        self.__printTime__(t0)

      return accuracy_per_batch
    
    # Run LwF
    def runLwF(self,train_batches,test_batches,net,fineTune=False):
      t0 = time.time()
      accuracy_per_batch = []
      for idx, batch in enumerate(train_batches):
        print(f'\n##### BATCH {idx+1} #####')
        n_classes = (idx+1)*10
        net = self.__updateRepresentation__(batch,{},net,n_classes,fineTune)
        self.__printTime__(t0)

        accuracy_per_batch.append(self.__FCClassifier__(test_batches[idx],net,n_classes))
        self.__printTime__(t0)

      return accuracy_per_batch
