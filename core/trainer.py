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
from network.network import get_model
from tqdm import tqdm
from pathlib import Path


def train_batch(data_batch, label_batch, model):
    if cfg.NUM_GPUS > 0 and torch.cuda.is_available():
        data_batch = data_batch.cuda()
        label_batch = label_batch.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.LEARN_RATE, momentum=cfg.MOMENTUM)
    pred_batch = model(data_batch)
    mse_loss = torch.nn.MSELoss()
    loss_batch = mse_loss(pred_batch, label_batch)

    optimizer.zero_grad()
    loss_batch.backward()
    optimizer.step()

    if cfg.NUM_GPUS > 0 and torch.cuda.is_available():
        loss_batch = loss_batch.cpu()
    return loss_batch


def train_model():
    out_path = Path(cfg.OUT_DIR)

    data_loader = get_data_loader()
    model = get_model()
    for i in range(cfg.MAX_EPOCH):
        epoch_loss = 0
        epoch_iter_num = 0
        for idx, item in enumerate(tqdm(data_loader)):
            loss_batch = train_batch(item['image_a'], item['image_b'], model)
            epoch_loss += loss_batch.detach().numpy()
            epoch_iter_num += 1
        print("Epoch: %02d/%02d - loss=%.4f" % (i, cfg.MAX_EPOCH, epoch_loss / epoch_iter_num))

        out_path.mkdir(exist_ok=True, parents=True)
        torch.save(model, out_path / Path('model_at_%d' % i))
    pass



def test_model():
    pass
