import os
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
import numpy as np
import random
import utils


class DataSet300W(Dataset):
    def __init__(self, data_path, apply_augment=True) -> None:
        super().__init__()

        self.apply_augment = apply_augment

        imgs_dir_path = os.path.join(data_path, "imgs")
        pts_dir_path = os.path.join(data_path, "pts")

        # self.datasetlist={'data':{},'label':{}}
        self.datasetlist = {"data": [], "label": []}

        imgs_dir, pts_dir = os.listdir(imgs_dir_path), os.listdir(pts_dir_path)

        if len(imgs_dir) != len(pts_dir):
            raise RuntimeError("imgs dir and pts dir do not have the same number of files")

        self.datanum = len(imgs_dir)

        for img, pt in zip(imgs_dir, pts_dir):
            if img != pt:
                raise RuntimeError("point and image do not match")
            # self.datasetlist['data'][img] = os.path.join(imgs_dir_path, img)
            # self.datasetlist['label'][pt] = os.path.join(pts_dir_path, pt)
            self.datasetlist["data"].append(os.path.join(imgs_dir_path, img))
            self.datasetlist["label"].append(os.path.join(pts_dir_path, pt))

        # img = np.load(self.datasetlist['data'][0])/255
        # point = np.load(self.datasetlist['label'][0])
        # img, point = self.augment(img, point)
        # utils.display_data(img, point)

    def __len__(self):
        return self.datanum

    def __getitem__(self, index):
        img_path = self.datasetlist["data"][index]
        points_path = self.datasetlist["label"][index]

        img = np.load(img_path).astype(np.float32)
        points = np.load(points_path).astype(np.float32)

        if self.apply_augment:
            img, points = self.augment(img, points)

        img = torch.from_numpy(np.ascontiguousarray(img))
        points = torch.from_numpy(np.ascontiguousarray(points))

        return img, points

    def augment(self, img, points):

        # horizontal flip
        if random.randint(0, 4) == 0:
            img = img[:, :, ::-1]
            points[:, 0] = 1 - points[:, 0]

        # vertical flip
        if random.randint(0, 4) == 0:
            img = img[:, ::-1, :]
            points[:, 1] = 1 - points[:, 1]

        # to grayscale
        if random.randint(0, 4) == 0:
            coefficients = np.array([0.2989, 0.5870, 0.1140]).reshape(3, 1, 1)
            img = np.sum(img * coefficients, axis=0)
            img = np.stack([img, img, img], axis=0)

        return img, points


# test = DataSet300W("./Face Detection/Data/lfpw/testset_npy")
# loader = DataLoader(test, batch_size=1)
# for x, y in loader:
#     print(x.shape)
#     exit()
