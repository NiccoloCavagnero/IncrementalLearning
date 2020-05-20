import torch
import torch.nn as nn
from copy import deepcopy

class LWFaNaNet(nn.Module):

    def __init__(self, EPOCHS, LR, NUM_CLASSES, TASK_SIZE, DEVICE, old_model = None):
        super(LWFaNaNet, self).__init__()

        # Parameters of net
        self.EPOCHS=EPOCHS
        self.LR=LR
        self.NUM_CLASSES = NUM_CLASSES
        self.TASK_SIZE = TASK_SIZE
        self.DEVICE = DEVICE
        self.old_model = old_model
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, NUM_CLASSES),
        )    
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def updateNet(net, n_classes):
    in_features = net.classifier[7].in_features
    out_features = net.classifier[7].out_features
    weight = net.classifier[7].weight.data
    bias = net.classifier[7].bias.data

    net.classifier[7] = nn.Linear(in_features, n_classes)
    net.classifier[7].weight.data[:out_features] = weight
    net.classifier[7].bias.data[:out_features] = bias

def LWFananet(NUM_EPOCHS, LR, NUM_CLASSES, TASK_SIZE, DEVICE, progress=True):
    model = LWFaNaNet(NUM_EPOCHS, LR, NUM_CLASSES, TASK_SIZE, DEVICE)
    return model
