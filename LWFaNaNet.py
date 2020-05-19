import torch
import torch.nn as nn
from copy import deepcopy

class aNaNet(nn.Module):

    def __init__(self, EPOCHS, LR, NUM_CLASSES, TASK_SIZE):
        super(aNaNet, self).__init__()

        # Parameters of net
        self.EPOCHS=EPOCHS
        self.LR=LR
        self.exemplar_set = []
        self.class_mean_set = []
        self.NUM_CLASSES = NUM_CLASSES
        self.TASK_SIZE = TASK_SIZE
        
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
            nn.Linear(4096, num_classes),
        )    
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def _updateNet_(self,net,n_classes):
        in_features = net.classifier[7].in_features
        out_features = net.classifier[7].out_features
        weight = net.classifier[7].weight.data
        bias = net.classifier[7].bias.data

        net.classifier[7] = nn.Linear(in_features, n_classes)
        net.classifier[7].weight.data[:out_features] = weight
        net.classifier[7].bias.data[:out_features] = bias

        return net


def LWFananet(progress=True, **kwargs):
    model = aNaNet(**kwargs)
    return model

net = ananet()
