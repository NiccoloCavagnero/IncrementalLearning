from torchvision.datasets import VisionDataset, CIFAR100
from torchvision import transforms
import numpy as np

class Cifar100(VisionDataset):
    def __init__(self, root, train, transform=None, target_transform=None):
        super(Cifar100, self).__init__(root, transform=transform, target_transform=target_transform)
        self.dataset = CIFAR100(root=root, train=train, download=True, transform=transform)
        
        shuffled_classes = [76, 53, 69, 24, 62, 61, 91,  1,  7, 31, 28, 26, 27, 43, 10, 83, 19,
        3, 54, 46, 13, 55, 85, 74, 94, 34, 73,  0, 71, 87, 20, 75, 22, 33,
        40, 84, 18, 64, 16, 14, 37, 12,  8, 77, 90,  6, 86, 63, 50, 21, 70,
        2, 41, 56, 89, 51, 72,  5, 45, 58, 67, 17, 49, 60, 95, 65, 99, 38,
        97, 23,  9, 39, 47, 52, 92, 88, 44, 78, 57, 35, 96, 29, 11, 81, 82,
        66, 42, 25, 68, 98, 36, 59,  4, 80, 93, 48, 15, 30, 32, 79]
        
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
