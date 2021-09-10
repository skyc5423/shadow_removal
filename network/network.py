'''
File: network.py
Project: skeleton
File Created: 2021-09-09 16:40:17 am
Author: sangmin.lee
-----
This script ...

Reference
...
'''

import torch
from cfg.config import cfg


def get_model():
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=False)

    if cfg.NUM_GPUS > 0 and torch.cuda.is_available():
        return model.cuda()
    else:
        return model
