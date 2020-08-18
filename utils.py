import numpy as np
import random
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix, accuracy_score

from matplotlib import pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

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
  
def confusionMatrix(labels, predictions, step):
  ticks = np.arange(10,110,10)      
  plt.figure(figsize=(8,8))
  cm = confusion_matrix(labels, predictions)
  sns.heatmap(np.log(cm+1),cmap='jet',cbar=False)
  plt.xticks(ticks[:step+1],labels=ticks[:step+1],rotation='horizontal')
  plt.yticks(ticks[:step+1],labels=ticks[:step+1],rotation='horizontal')
  plt.xlabel('Predicted Class')
  plt.ylabel('True Class')
  plt.show()

 
def accuracyPlot(accuracies, std, names, title):

    fig = go.Figure()
    for idx, el in enumerate(names):

        print(accuracies[idx])
        fig.add_trace(go.Scatter(
            x=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            y=accuracies[idx],
            error_y=dict(
                type='data',
                array=std[idx]
            ),
            name=el
        ))

    array = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for i in array:
        fig.add_shape(
            dict(
                type="line",
                x0=0,
                y0=i,
                x1=100,
                y1=i,
                line=dict(
                    color="Grey",
                    width=1,
                    dash="dot",
                )
            ))
fig['layout']['yaxis'].update(title='Accuracy', range=[0, 1], dtick=0.1, tickcolor='black', ticks="outside",
                                  tickwidth=1, ticklen=5)
    fig['layout']['xaxis'].update(title='Number of classes', range=[0, 100.5], dtick=10, ticks="outside", tickwidth=0)
    fig['layout'].update(height=900, width=900)
    fig['layout'].update(plot_bgcolor='rgb(256,256,256)')
    fig.show()
  
def printTime(t0):
  print(f'\n   # Elapsed time: {round((time.time()-t0)/60,2)}')
  
def CELoss(outputs,targets):
  logsoftmax = torch.nn.LogSoftmax()
  return torch.mean(torch.sum(- targets * logsoftmax(outputs),1))

############################ CLASSIFIERS #################################
  
def NMEClassifier(data,batch,exemplars,net,n_classes,device):
  print(f'\n ### NME ###')
  means = dict.fromkeys(np.arange(n_classes))
  net.eval()

  batch_map = fillClassMap(batch,n_classes)

  print('   # Computing means ')
  for key in range(n_classes):
    if key in range(n_classes-10,n_classes):
      items = batch_map[key]
    else:
      items = exemplars[key]
        
    loader = DataLoader(items, batch_size=512, shuffle=False, num_workers=4, drop_last=False)
    mean = torch.zeros((1,64),device=device)
    for images, _ in loader:
      with torch.no_grad():
        images = images.to(device)
        flipped_images = torch.flip(images,[3])
        images = torch.cat((images,flipped_images))
        
        outputs = net(images,features=True)
            
        for output in outputs:
          mean += output
    mean = mean / ( 2 * len(items) ) 
    means[key] = mean / mean.norm()

  loader = DataLoader(data, batch_size=512, shuffle=False, num_workers=4, drop_last=False)

  predictions, label_list = [], []
  print('   # NME Predicting ')
  for images, labels in loader:
    images = images.to(device)
    label_list += list(labels)
    with torch.no_grad():
      outputs = net(images,features=True)
      for output in outputs:
        prediction = None
        min_dist = 99999
        for key in means:
          dist = torch.dist(means[key],output)
          if dist < min_dist:
            min_dist = dist
            prediction = key
        predictions.append(prediction)
  
  accuracy = accuracy_score(label_list,predictions)
  print(f'   # NME Accuracy: {accuracy}')

  return accuracy, predictions, label_list

def FCClassifier(data,net,n_classes,device):
  print(f'\n ### FC Layer ###')
  print('   # FC Layer Predicting ')
  net.eval()
      
  running_corrects = 0.0
  label_list, predictions = [], []
  with torch.no_grad():
    loader = DataLoader(data, batch_size=512, shuffle=False, num_workers=4, drop_last=False)
    for images, labels in loader:
      images = images.to(device)
      labels = labels.to(device)

      outputs = torch.sigmoid(net(images))
      # Get predictions
      _, preds = torch.max(outputs.data, 1)
      # Update Corrects
      running_corrects += torch.sum(preds == labels.data).data.item()
          
      for prediction,label in zip(preds,labels):
        predictions.append(np.array(prediction.cpu()))
        label_list.append(np.array(label.cpu()))

  # Compute Accuracy
  accuracy = running_corrects / len(data)
      
  print(f'   # FC Layer Accuracy: {accuracy}')
  return accuracy, predictions, label_list
    
############################ EXEMPLARS #################################
    
def randomExemplarSet(memory,data,n_classes):
  print('\n ### Construct Random Exemplar Set ###')
  if n_classes != 10:
    m = int(memory/(n_classes-10))
  else:
    m = int(memory/(n_classes))
  print(f'   # Exemplars per class: {m}')

  # Initialize lists of images and exemplars for each class
  class_map = fillClassMap(data,n_classes)
  exemplars = dict.fromkeys(np.arange(n_classes-10,n_classes))
  for label in exemplars:
    exemplars[label] = []

  for label in class_map:
    indexes = random.sample(range(len(class_map[label])),m)   
    for idx in indexes:
      exemplars[label].append(class_map[label][idx])
  
  return exemplars
  
def reduceExemplarSet(memory,exemplars,n_classes):
  print('\n ### Reduce Exemplar Set ###')
  m = int(memory/n_classes)
  print(f'   # Exemplars per class: {m}')
  for key in exemplars:
    exemplars[key] = exemplars[key][:m]
      
  return exemplars
    
# dict to list
def formatExemplars(exemplars):
  new_exemplars = []
  for key in exemplars:
    for item in exemplars[key]:
      new_exemplars.append([item[0],item[1]])

  return new_exemplars
    
