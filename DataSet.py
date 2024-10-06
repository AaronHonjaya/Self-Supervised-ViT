import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import numpy as np
import random
import abc
import natsort 
import torchvision.transforms as transforms


class BaseDataset(Dataset, abc.ABC):
    def __init__(self, data_path, patch_size, stride = None, apply_augment=False, generator = None) -> None:
        super().__init__()

        self.datapath = data_path
        self.apply_augment = apply_augment
        self.generator = generator
        
        self.patch_size = patch_size
        self.stride = patch_size if stride is None else stride
        
        self.wnid_to_words = {}
        with open("./tiny-imagenet-200/classes.txt", 'r+') as file:
            for line in file:
                splits = line.split("\t")
                self.wnid_to_words[splits[0]] = splits[1].strip().split(", ")


    @abc.abstractmethod 
    def _get_image_path(self, index):
        pass
    
    @abc.abstractmethod 
    def _get_label(self, index):
        pass
    
    
    def __len__(self):
        return self.datanum
        
    def __getitem__(self, index):
        img_path = self._get_image_path(index)

        img = np.load(img_path).astype(np.float32)

        img = torch.from_numpy(np.ascontiguousarray(img))
        label = self._get_label(index)

        # if self.apply_augment:
        img = self.augment(img)
            
        img = F.unfold(img, kernel_size=self.patch_size, stride=self.stride)
                
        return img.swapaxes(0, 1), label

    def augment(self, img):

        # horizontal flip
        if random.randint(0, 4) == 0:
            img = torch.flip(img, [2])

        # vertical flip
        if random.randint(0, 4) == 0:

            img = torch.flip(img, [1])

        # to grayscale
        if random.randint(0, 4) == 0:
            to_grayscale = transforms.Grayscale(3)
            img = to_grayscale(img)
            
        return img
    
class TrainDatasetImageNet(BaseDataset):
    def __init__(self, train_dir, patch_size, stride = None, apply_augment=False, generator = None) -> None:
        super().__init__(train_dir, patch_size, stride, apply_augment, generator)

        self.wnids = []
        for subdir in natsort.natsorted(os.listdir(train_dir)):
            self.wnids.append(subdir)
            
            # make sure exactly 500 images per class. 
            assert len(os.listdir(os.path.join(train_dir, subdir, "images_npy"))) == 500
        
        
        self.images_per_class = 500
        self.datanum = 500 * len(subdir)
        

    def _get_image_path(self, index):
        img_num = index % self.images_per_class
        wnid = self.wnids[index//self.images_per_class]
        path = os.path.join(self.datapath, wnid,"images_npy", f'{wnid}_{img_num}.npy')
        return path

    def _get_label(self, index):
        return torch.tensor(index // self.images_per_class, dtype=torch.int32)


# train = TrainDatasetImageNet("./tiny-imagenet-200/train", 8)
# loader = DataLoader(train, batch_size=1)

# for i, (x, y) in enumerate(loader):
#     print(x.shape)
#     print(y)
#     exit()
