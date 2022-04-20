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
    parser.add_argument('--batch-size', type=int, default=4, help="batch size")
    parser.add_argument('--num-workers', type=int, default=8, help="batch size")
    parser.add_argument('--end-epoch', type=int, default=1000, help="maxmium number of training epoch")
    parser.add_argument('--resume-epoch', type=int, default=-1, help="resume train")
    parser.add_argument('--random-seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--gpus', type=list, default=[0,1], help='GPU to use')
    parser.add_argument('--network', type=str, default='R3D18',help="chose the base network")


    