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
        self.pixel_mean = self.__getPixelMean__(root)
        
        shuffled_classes = [14, 69, 22, 28, 55, 33, 85, 51, 8, 99, 94, 77, 90, 84, 35, 65, 60, 52,
                            0, 87, 59, 27, 92, 6, 70, 79, 21, 16, 47, 62, 3, 9, 49, 5, 44, 4, 57,
                            82, 74, 15, 89, 19, 48, 95, 50, 10, 7, 67, 46, 36, 34, 38, 53, 54, 23,
                            83, 37, 18, 39, 78, 73, 86, 96, 1, 58, 98, 20, 12, 97, 11, 29, 75, 81,
                            42, 56, 88, 24, 76, 66, 61, 31, 72, 13, 43, 2, 25, 40, 64, 71, 26, 93,
                            17, 63, 30, 45, 41, 80, 68, 91, 32]
        
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
            image -= self.pixel_mean
            
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
    
    def __getPixelMean__(self,root):
        dataset = CIFAR100(root=root,train=True,transform=transforms.ToTensor())
        mean = torch.zeros((3,32,32))
        loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=4, drop_last=False)
        for images, _ in loader:
            mean += sum(images)
        return mean / 50000
