from torchvision.datasets import VisionDataset, CIFAR100
from PIL import Image
import os
import numpy as np

class Cifar100(VisionDataset):
    def _init_(self, root, train, transform=None, target_transform=None):
        super(Cifar100, self)._init_(root, transform=transform, target_transform=target_transform)
        self.dataset = CIFAR100(root=root, train=train, download=True, transform=transform)
        self.transform = transform

    def _getitem_(self, index):
        image, label = self.dataset[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def _len_(self):
        return len(self.dataset)

    # Returns indexes of images whose label is
    # in classes (-> Subset method)
    def _getClassBatch_(self,classes):
        classes = set(classes)
        batch = []
        for idx,item in enumerate(self.dataset):
            if item[1] in classes:
                batch.append(idx)
        return batch
