'''
File: istd_dataset.py
Project: sahadow_removal
File Created: 2021-08-29 15:32:48 am
Author: sangmin.lee
-----
This script ...

Reference
...
'''
from torch.utils.data import Dataset
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import os
import random
from PIL import Image
from pathlib import Path
from cfg.config import cfg


class ISTDDataSet(Dataset):

    def __init__(self, data_path):
        self.data_path_a = Path(data_path % 'A')
        self.data_path_b = Path(data_path % 'B')
        self.data_path_c = Path(data_path % 'C')
        self.list_file_name = os.listdir(self.data_path_a)

    def __len__(self):
        return len(self.list_file_name)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path_a = self.data_path_a / Path(self.list_file_name[idx])
        image_a = Image.open(image_path_a)
        image_path_b = self.data_path_b / Path(self.list_file_name[idx])
        image_b = Image.open(image_path_b)
        image_path_c = self.data_path_c / Path(self.list_file_name[idx])
        image_c = Image.open(image_path_c)

        # Random horizontal flipping
        if random.random() > 0.5:
            image_a = TF.hflip(image_a)
            image_b = TF.hflip(image_b)
            image_c = TF.hflip(image_c)

        # Random vertical flipping
        if random.random() > 0.5:
            image_a = TF.hflip(image_a)
            image_b = TF.hflip(image_b)
            image_c = TF.hflip(image_c)

        # Transform to tensor
        image_a = TF.to_tensor(image_a)
        image_b = TF.to_tensor(image_b)
        image_c = TF.to_tensor(image_c)

        return {'image_a': image_a, 'image_b': image_b, 'image_c': image_c}


def get_data_loader():
    dataset = ISTDDataSet('./data/ISTD_Dataset/train/train_%s')
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=cfg.DATA_LOADER.TRAIN_BATCH,
                                              shuffle=True,
                                              num_workers=cfg.DATA_LOADER.NUM_WORKERS)

    return data_loader
