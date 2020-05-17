from torchvision.datasets import VisionDataset, CIFAR100
from torchvision import transforms
import numpy as np

class Cifar100(VisionDataset):
    def __init__(self, root, train, transform=None, target_transform=None):
        super(Cifar100, self).__init__(root, transform=transform, target_transform=target_transform)
        self.dataset = CIFAR100(root=root, train=train, download=True, transform=transform)
        
        shuffled_classes = [53, 36, 38, 42, 80, 45, 11, 28, 54, 86, 7, 0, 89,
                    21, 18, 35, 1, 47, 92, 31, 52, 85, 63, 12, 66, 75,
                    43, 24, 77, 94, 79, 25, 73, 8, 6, 46, 78, 69, 2, 84,
                    17, 48, 9, 44, 50, 61, 41, 39, 26, 71, 90, 5, 97,
                    96, 74, 87, 27, 51, 98, 19, 32, 55, 40, 76, 23, 33,
                    59, 49, 67, 58, 93, 95, 83, 16, 65, 20, 30, 72, 56,
                    91, 62, 13, 3, 82, 10, 34, 88, 37, 60, 14, 57, 4,
                    15, 29, 64, 70, 99, 68, 22, 81]
        # Define classes per batch
        self.class_batches = dict.fromkeys(np.arange(10))
        for i in range(10):
            self.class_batches[i] = shuffled_classes[i*10:(i*10+10)]
        
        # Dictionary key:batch, value:batch_indexes
        self.batch_indexes = self.__BatchIndexes__()

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label

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
