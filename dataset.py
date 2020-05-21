from torchvision.datasets import VisionDataset, CIFAR100
from torchvision import transforms
import numpy as np

class Cifar100(VisionDataset):
    def __init__(self, root, train, transform=None, target_transform=None):
        super(Cifar100, self).__init__(root, transform=transform, target_transform=target_transform)
        self.dataset = CIFAR100(root=root, train=train, download=True, transform=transform)
        
        shuffled_classes = [43,  1, 22,  4, 62, 92, 14, 18, 49, 84, 78, 56, 26, 93,  8, 89,  0,
       70, 86, 39, 23, 13, 94, 61, 31, 74,  6, 69, 72, 83, 15, 55, 63, 59,
       27, 30, 38, 35, 46, 65, 20, 87, 16, 41, 96, 53,  9, 40,  2, 42, 58,
       60, 95, 47, 77, 76, 79, 91, 68, 66, 73, 98, 51, 48, 25, 11, 54, 97,
       90, 88,  3, 33, 17, 85, 37, 29, 28, 50, 10, 19,  5, 67, 45, 34, 44,
       32, 82, 75, 24, 81, 52, 71, 21, 57, 36, 80, 99, 64,  7, 12]
        
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
        return image, self.label_map[label], index

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
