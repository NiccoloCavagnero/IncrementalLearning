from torchvision.datasets import VisionDataset, CIFAR100

class Cifar100(VisionDataset):
    def __init__(self, root, train, transform=None, target_transform=None):
        super(Cifar100, self).__init__(root, transform=transform, target_transform=target_transform)
        self.dataset = CIFAR100(root=root, train=train, download=True, transform=transform)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label

    def __len__(self):
        return len(self.dataset)

    def __getClassBatch__(self,classes):
        '''
        Returns indexes of images whose label is
        in classes (-> Subset method)
        '''
        classes = set(classes)
        batch = []
        for idx,item in enumerate(self.dataset):
            if item[1] in classes:
                batch.append(idx)
        return batch
