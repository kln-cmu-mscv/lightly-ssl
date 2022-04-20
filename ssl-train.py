import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import lightly
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from PIL import Image
import numpy as np
import torchvision
import dataset
import argparse
from data.iterator_factory import get_arid
from data.iterator_factory import ARIDCollateFunction
from network.symbol_builder import get_symbol
from network.simclr import SimCLR_R3D_Model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Video Classification Parser")
    parser.add_argument('--dataset', default='ARID', help="path to dataset")
    parser.add_argument('--data-root', default="./dataset/ARID", help="path to dataset")
    parser.add_argument('--clip-length', type=int, default=16, help="define the length of each input sample.")
    parser.add_argument('--train-frame-interval', type=int, default=2, help="define the sampling interval between frames.")
    parser.add_argument('--batch-size', type=int, default=8, help="batch size")
    parser.add_argument('--num-workers', type=int, default=8, help="batch size")
    parser.add_argument('--end-epoch', type=int, default=2, help="maxmium number of training epoch")
    parser.add_argument('--resume-epoch', type=int, default=-1, help="resume train")
    parser.add_argument('--random-seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--gpus', type=str, default="0", help='GPU to use')
    parser.add_argument('--network', type=str, default='R3D18',help="chose the base network")

    # set args
    args = parser.parse_args()

    pl.seed_everything(args.random_seed)

    iter_seed = torch.initial_seed() + 100 + max(0, args.resume_epoch) * 100

    # load dataset related configuration
    dataset_cfg = dataset.get_config(name=args.dataset)
    dataset_cfg['num_classes'] = 512 #Setting last fc layer output size = 512

    # create model with all parameters initialized
    net, input_conf = get_symbol(name=args.network, pretrained=True, **dataset_cfg)

    # create dataset 
    train_dataset = get_arid(name=args.dataset, 
                            clip_length=args.clip_length, 
                            train_interval=args.train_frame_interval,
                            mean=input_conf['mean'], std=input_conf['std'], 
                            seed=iter_seed, 
                            data_root=args.data_root,
                            return_item_subpath=True)

    # define collate function with different augmentation
    collate_fn = ARIDCollateFunction()

    # define train dataloader
    dataloader_train_arid = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=args.num_workers
    )

    max_epochs = args.end_epoch
    model = SimCLR_R3D_Model(net, max_epochs)
    trainer = pl.Trainer(
        max_epochs=max_epochs, gpus=args.gpus, progress_bar_refresh_rate=100
    )
    trainer.fit(model, dataloader_train_arid)