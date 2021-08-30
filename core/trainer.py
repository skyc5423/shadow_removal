'''
File: trainer.py
Project: skeleton
File Created: 2021-08-30 01:05:15 am
Author: sangmin.lee
-----
This script ...

Reference
...
'''
from cfg.config import cfg
from data.istd_dataset import get_data_loader
import torch


def train_model():
    data_loader = get_data_loader()
    for idx, item in enumerate(data_loader):
        print()
    pass


def test_model():
    pass
