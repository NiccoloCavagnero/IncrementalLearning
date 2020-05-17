from torchvision.datasets import VisionDataset, CIFAR100
from torch.utils.data import Subset

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
        self.batches = dict.fromkeys([x for x in range(10)])
        for batch in self.batches:
            self.batches[batch] = shuffled_classes[batch:(batch+10)]

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label

    def __len__(self):
        return len(self.dataset)

    def __getClassIndexes__(self,classes):
        '''
        Returns indexes of images whose label is
        in classes (-> Subset method)
        '''
        classes = set(classes)
        indexes = []
        for idx,item in enumerate(self.dataset):
            if item[1] in classes:
                indexes.append(idx)
        return indexes

    def __getClassBatch__(self,batch_index):
        indexes = self.__getClassIndexes__(self.batches[batch_index])
        return Subset(self.dataset,indexes)
