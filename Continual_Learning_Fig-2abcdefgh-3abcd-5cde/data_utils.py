import numpy as np
import pandas as pd
import torch
import torchvision
import matplotlib.pyplot as plt
import os
import json

from collections import OrderedDict
from datetime import datetime
from PIL import Image


transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=(0.0,), std=(1.0,))])

mnist_dset_train = torchvision.datasets.MNIST('./mnist_pytorch', train=True, transform=transform, target_transform=None, download=True)
mnist_train_loader = torch.utils.data.DataLoader(mnist_dset_train, batch_size=100, shuffle=True, num_workers=1)

mnist_dset_test = torchvision.datasets.MNIST('./mnist_pytorch', train=False, transform=transform, target_transform=None, download=True)
mnist_test_loader = torch.utils.data.DataLoader(mnist_dset_test, batch_size=100, shuffle=False, num_workers=1)

fmnist_dset_train = torchvision.datasets.FashionMNIST('./fmnist_pytorch', train=True, transform=transform, target_transform=None, download=True)
fashion_mnist_train_loader = torch.utils.data.DataLoader(fmnist_dset_train, batch_size=100, shuffle=True, num_workers=1)

fmnist_dset_test = torchvision.datasets.FashionMNIST('./fmnist_pytorch', train=False, transform=transform, target_transform=None, download=True)
fashion_mnist_test_loader = torch.utils.data.DataLoader(fmnist_dset_test, batch_size=100, shuffle=False, num_workers=1)

usps_transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize(28,28), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=(0.0,), std=(1.0,))])

usps_dset_train = torchvision.datasets.USPS('./usps_pytorch', train=True, transform=usps_transform, target_transform=None, download=True)
usps_train_loader = torch.utils.data.DataLoader(usps_dset_train, batch_size=100, shuffle=True, num_workers=1)

usps_dset_test = torchvision.datasets.USPS('./usps_pytorch', train=False, transform=usps_transform, target_transform=None, download=True)
usps_test_loader = torch.utils.data.DataLoader(usps_dset_test, batch_size=100, shuffle=False, num_workers=1)

def create_permuted_loaders(task):
    
    permut = torch.from_numpy(np.random.permutation(784))
        
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                      torchvision.transforms.Lambda(lambda x: x.view(-1)[permut].view(1, 28, 28) ),
                      torchvision.transforms.Normalize(mean=(0.0,), std=(1.0,))])

    if task=='MNIST':
        dset_train = torchvision.datasets.MNIST('./mnist_pytorch', train=True, transform=transform, target_transform=None, download=True)
        train_loader = torch.utils.data.DataLoader(dset_train, batch_size=100, shuffle=True, num_workers=1)

        dset_test = torchvision.datasets.MNIST('./mnist_pytorch', train=False, transform=transform, target_transform=None, download=True)
        test_loader = torch.utils.data.DataLoader(dset_test, batch_size=1000, shuffle=False, num_workers=1)

        
    elif task=='FMNIST':
   
        dset_train = torchvision.datasets.FashionMNIST('./fmnist_pytorch', train=True, transform=transform, target_transform=None, download=True)
        train_loader = torch.utils.data.DataLoader(dset_train, batch_size=100, shuffle=True, num_workers=1)

        dset_test = torchvision.datasets.FashionMNIST('./fmnist_pytorch', train=False, transform=transform, target_transform=None, download=True)
        test_loader = torch.utils.data.DataLoader(dset_test, batch_size=100, shuffle=False, num_workers=1)

    return train_loader, test_loader, dset_train



def createHyperparametersFile(path, args):

    hyperparameters = open(path + r"/hyperparameters.txt","w+")
    L = ["- scenario: {}".format(args.scenario) + "\n",
        "- interleaved: {}".format(args.interleaved) + "\n",
        "- hidden layers: {}".format(args.hidden_layers) + "\n",
        "- normalization: {}".format(args.norm) + "\n",
        "- net: {}".format(args.net) + "\n",
        "- task sequence: {}".format(args.task_sequence) + "\n",
        "- lr: {}".format(args.lr) + "\n",
        "- gamma: {}".format(args.gamma) + "\n",
        "- meta: {}".format(args.meta) + "\n",
        "- beaker: {}".format(args.beaker) + "\n",
        "- number of beakers: {}".format(args.n_bk) + "\n",
        "- ratios: {}".format(args.ratios) + "\n",
        "- feedback: {}".format(args.fb) + "\n",
        "- ewc: {}".format(args.ewc) + "\n",
        "- ewc lambda: {}".format(args.ewc_lambda) + "\n",
        "- SI: {}".format(args.si) + "\n",
        "- Binary Path Integral: {}".format(args.bin_path) + "\n",
        "- SI lambda: {}".format(args.si_lambda) + "\n",
        "- decay: {}".format(args.decay) + "\n",
        "- epochs per task: {}".format(args.epochs_per_task) + "\n",
        "- init: {}".format(args.init) + "\n",
        "- init width: {}".format(args.init_width) + "\n",
        "- seed: {}".format(args.seed) + "\n"]
   
    hyperparameters.writelines(L)
    hyperparameters.close()
        







