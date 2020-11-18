import os
import logging
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from dataset import DataTrain
from model_define import UNet
from solver import Solver

args = {
    'batch_size': 32,
    'log_interval': 1,
    'log_dir': 'log',
    'num_classes': 18,
    'epochs': 1000,
    'lr': 1e-6,
    'resume': True,
    'data_dir': "../data",
    'gamma': 0.5,
    'step': 10
}
'''
文件目录：
data
    images
        *.tif
    labels
        *.png
code
    train.py
    ...
'''


def log_init():
    if not os.path.exists(args['log_dir']):
        os.makedirs(args['log_dir'])
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(
        logging.FileHandler(os.path.join(args['log_dir'], "log.txt")))
    logger.info("%s", repr(args))
    return logger


def train():
    logger = log_init()

    dataset_train = DataTrain(args['data_dir'])
    train_dataloader = DataLoader(dataset=dataset_train,
                                  batch_size=args['batch_size'],
                                  shuffle=False)
    model = UNet(3, 18)
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.cuda()

    solver = Solver(num_classes=args['num_classes'],
                    lr_args={
                        "gamma": args['gamma'],
                        "step_size": args['step']
                    },
                    optimizer_args={
                        "lr": args['lr'],
                        "betas": (0.9, 0.999),
                        "eps": 1e-8,
                        "weight_decay": 0.01
                    },
                    optimizer=torch.optim.Adam)
    solver.train(model,
                 train_dataloader,
                 num_epochs=args['epochs'],
                 log_params={
                     'logdir': args['log_dir'] + "/logs",
                     'log_iter': args['log_interval'],
                     'logger': logger
                 },
                 expdir=args['log_dir'] + "/ckpts",
                 resume=args['resume'])


if __name__ == "__main__":
    train()
