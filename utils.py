import numpy as np
import time
import torch
from torch.nn import functional as F

from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt

def updateNet(net, n_classes):
  in_features = net.fc.in_features
  out_features = net.fc.out_features
  weight = net.fc.weight.data
  bias = net.fc.bias.data

  net.fc = torch.nn.Linear(in_features, n_classes)
  net.fc.weight.data[:out_features] = weight
  net.fc.bias.data[:out_features] = bias

  return net
  
def getOneHot(target, n_classes):
  one_hot = torch.zeros(target.shape[0], n_classes)
  one_hot = one_hot.scatter(dim=1,index=target.long().view(-1,1),value=1.)
      
  return one_hot
  
def augmentation(image):
  image = F.pad(image,(4,4,4,4),value=0)
  x,y = np.random.randint(8),np.random.randint(8)
  image = image[:,x:x+32,y:y+32]
  if np.random.randint(2):
    image = torch.flip(image,[2])
    
  return image
  
def fillClassMap(data, n_classes):
  class_map = dict.fromkeys(np.arange(n_classes-10,n_classes))  
  for label in class_map:
    class_map[label] =  []
        
  # Fill class_map
  for item in data:
    for label in class_map:
      if item[1] == label:
        class_map[label].append(item)

  return class_map
  
def confusionMatrix(labels, predictions):
  cm = confusion_matrix(labels, predictions)
  sns.heatmap(np.log(cm+1),cmap='jet',cbar=False)
  plt.xlabel('Predicted Class')
  plt.ylabel('True Class')
  plt.show()
  
def printTime(t0):
  print(f'\n   # Elapsed time: {round((time.time()-t0)/60,2)}')
    
