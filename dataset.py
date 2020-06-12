from torchvision.datasets import VisionDataset, CIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import numpy as np

class Cifar100(VisionDataset):
    def __init__(self, root, train, transform=None, target_transform=None):
        super(Cifar100, self).__init__(root, transform=transform, target_transform=target_transform)
        self.dataset = CIFAR100(root=root, train=train, download=True, transform=None)
        self.transform = transform
        '''
        #best
        shuffled_classes = [61, 34, 79, 90,  9, 17, 68, 54, 74, 99, 75, 46, 83, 57, 77, 28, 52,
        40, 93, 12, 82, 89, 19, 43, 95, 48, 85, 86,  0, 53, 58, 63, 65, 94,
        16, 36,  1, 23, 15, 24, 55, 31, 27, 81, 71, 84, 30, 44, 73, 42,  2,
        76, 92, 32, 87, 78, 13, 56, 38, 96, 18, 33, 67, 69,  4, 64,  8, 72,
        98,  3, 39, 60, 59,  7, 11, 51,  5, 49, 35, 45, 70, 88, 41, 37, 66,
        80, 21, 29,  6, 20, 62, 97, 25, 26, 47, 22, 14, 10, 50, 91]
        '''
        shuffled_classes = [97, 78, 96, 10, 25, 50, 92, 88, 29, 45, 66, 71, 38, 84, 43, 62, 32,
        16, 33, 87, 13, 70, 31, 7, 23, 11, 51, 67, 86, 99, 90, 48, 56, 0,
        74, 61, 21, 4, 57, 83, 77, 8, 28, 89, 46, 69, 2, 39, 94, 20, 53,
        1, 18,  9, 85, 81, 65, 36, 79, 95, 5, 75, 80, 22, 15, 55, 6, 41,
        73, 14, 59, 82, 42, 30, 3, 93, 35, 54, 12, 52, 98, 27, 47, 19, 24,
        63, 58, 91, 76, 60, 34, 44, 37, 68, 64, 40, 26, 72, 17, 49]
        
        # Define classes per batch
        self.class_batches = dict.fromkeys(np.arange(10))
        for i in range(10):
            self.class_batches[i] = shuffled_classes[i*10:(i*10+10)]
        
        # Dictionary key:batch, value:batch_indexes
        self.batch_indexes = self.__BatchIndexes__()
        
        # Map labels from 0 to 99
        self.label_map = {k: v for v, k in enumerate(shuffled_classes)}
      

    def __getitem__(self, index):
        image, label = self.dataset[index]
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, self.label_map[label]

    def __len__(self):
        return len(self.dataset)

    def __BatchIndexes__(self):
        batches = dict.fromkeys(np.arange(10))
        for i in range(10):
            batches[i] = []
        for idx,item in enumerate(self.dataset):
            for i in range(10):
                if item[1] in self.class_batches[i]:
                    batches[i].append(idx)
        return batches

    def __getBatchIndexes__(self,batch_index):
        return self.batch_indexes[batch_index]
