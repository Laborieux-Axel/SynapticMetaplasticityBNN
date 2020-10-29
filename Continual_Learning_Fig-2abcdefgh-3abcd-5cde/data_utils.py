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

#usps_transform = torchvision.transforms.Compose( [torchvision.transforms.Resize((28,28)),
#    torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=(0.0,), std=(1.0,))])

#usps_dset_train = torchvision.datasets.USPS('./usps_pytorch', train=True, transform=usps_transform, target_transform=None, download=True)
#usps_train_loader = torch.utils.data.DataLoader(usps_dset_train, batch_size=12, shuffle=True, num_workers=1)

#usps_dset_test = torchvision.datasets.USPS('./usps_pytorch', train=False, transform=usps_transform, target_transform=None, download=True)
#usps_test_loader = torch.utils.data.DataLoader(usps_dset_test, batch_size=100, shuffle=False, num_workers=1)

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


class DatasetProcessing(torch.utils.data.Dataset): 
    def __init__(self, data, target, transform=None): 
        self.transform = transform
        self.data = data.astype(np.float32)[:,:,None]
        self.target = torch.from_numpy(target).long()
    def __getitem__(self, index): 
        if self.transform is not None:
            return self.transform(self.data[index]), self.target[index]
        else:
            return self.data[index], self.target[index]
    def __len__(self): 
        return len(list(self.data))


def process_features(X_train, X_test, mode):
    if mode=="cutoff":
        cutoff = 8
        threshold_train = np.zeros((np.shape(X_train)[0],1))
        threshold_test = np.zeros((np.shape(X_test)[0],1)) 
        for i in range(np.shape(X_train)[0]):
            threshold_train[i,0] = np.unique(X_train[i,:])[-cutoff]
        for i in range(np.shape(X_test)[0]):
            threshold_test[i,0] = np.unique(X_test[i,:])[-cutoff]
        X_train =   (np.sign(X_train  - threshold_train + 1e-6 ) + 1.0)/2
        X_test =  (np.sign (X_test  - threshold_test +1e-6 ) + 1.0)/2
    elif mode=="mean_over_examples":
        X_train = ( X_train - X_train.mean(axis = 0, keepdims = True) )/ X_train.var(axis =0, keepdims = True) # ???
        X_test = ( X_test - X_test.mean(axis=0, keepdims = True) ) /X_test.var(axis = 0, keepdims = True)
    elif mode=="mean_over_examples_sign":
        X_train =   (np.sign(X_train  - X_train.mean(axis = 0, keepdims = True) ) + 1.0)/2
        X_test =  (np.sign (X_test  - X_test.mean(axis = 0, keepdims = True) ) + 1.0)/2
    elif mode=="mean_over_pixels":
        X_train = ( X_train - X_train.mean(axis = 1, keepdims = True) )/ X_train.var(axis =1, keepdims = True)  # Instance norm
        X_test = ( X_test - X_test.mean(axis=1, keepdims = True) ) /X_test.var(axis = 1, keepdims = True)
    elif mode=="mean_over_pixels_sign":
        X_train =   (np.sign(X_train  - X_train.mean(axis = 1, keepdims = True) ) + 1.0)/2  
        X_test =  (np.sign (X_test  - X_test.mean(axis = 1, keepdims = True) ) + 1.0)/2
    elif mode=="global_mean":
        X_train = ( X_train - X_train.mean(keepdims = True) )/ X_train.var(keepdims = True) # Batch norm
        X_test = ( X_test - X_test.mean(keepdims = True) ) /X_test.var(keepdims = True)
    elif mode=="rescale":
        X_train =  (X_train / X_train.max(axis = 1, keepdims = True) )
        X_test =  (X_test / X_test.max(axis = 1, keepdims = True) )
    return X_train, X_test


def relabel(label):
    label_map = [5,6,0,1,2,3,4,7,8,9]
    return label_map[label]

vrelabel = np.vectorize(relabel)


def process_cifar10(subset):

    cifar_X_train = torch.load('cifar10_features_dataset/train.pt').cpu().numpy()
    cifar_Y_train = torch.load('cifar10_features_dataset/train_targets.pt').cpu().numpy() 
    cifar_X_test = torch.load('cifar10_features_dataset/test.pt').cpu().numpy() 
    cifar_Y_test = torch.load('cifar10_features_dataset/test_targets.pt').cpu().numpy()

    cifar_Y_train = vrelabel(cifar_Y_train)
    cifar_Y_test = vrelabel(cifar_Y_test)

    if subset=='animals':
        partition = np.vectorize(lambda l: l < 5) 
    elif subset=='vehicles':
        partition = np.vectorize(lambda l: l >= 5)  
    else:
        raise('error unsuported subset')
 
    mode = 'mean_over_pixels'
    sub_X_train = cifar_X_train[partition(cifar_Y_train)] 
    sub_X_test = cifar_X_test[partition(cifar_Y_test)] 
 
    sub_X_train, sub_X_test = process_features(sub_X_train, sub_X_test, mode) 
 
    sub_Y_train = cifar_Y_train[partition(cifar_Y_train)] 
    sub_Y_test = cifar_Y_test[partition(cifar_Y_test)] 
 
    sub_dset_train = DatasetProcessing(sub_X_train, sub_Y_train) 
    sub_train_loader = torch.utils.data.DataLoader(sub_dset_train, batch_size=100, shuffle=True, num_workers=4) 
 
    sub_dset_test = DatasetProcessing(sub_X_test, sub_Y_test) 
    sub_test_loader = torch.utils.data.DataLoader(sub_dset_test, batch_size=100, shuffle=False, num_workers=0) 
 
    return sub_train_loader, sub_test_loader, sub_dset_train 




def process_cifar100(n_subset):
    subset_size = 100//n_subset

    train_loader_list = []
    test_loader_list = []
    dset_train_list = []

    cifar100_X_train = torch.load('cifar100_features_dataset/train.pt').cpu().numpy()
    cifar100_Y_train = torch.load('cifar100_features_dataset/train_targets.pt').cpu().numpy()
    cifar100_X_test = torch.load('cifar100_features_dataset/test.pt').cpu().numpy()
    cifar100_Y_test = torch.load('cifar100_features_dataset/test_targets.pt').cpu().numpy()

    for k in range(n_subset):
        partition = np.vectorize(lambda l: ((l < (k+1)*subset_size) and (l >= k*subset_size)) )
        mode = 'mean_over_pixels'
        sub_X_train = cifar100_X_train[partition(cifar100_Y_train)]
        sub_X_test = cifar100_X_test[partition(cifar100_Y_test)]

        sub_X_train, sub_X_test = process_features(sub_X_train, sub_X_test, mode)

        sub_Y_train = cifar100_Y_train[partition(cifar100_Y_train)]
        sub_Y_test = cifar100_Y_test[partition(cifar100_Y_test)]

        sub_dset_train = DatasetProcessing(sub_X_train, sub_Y_train)
        sub_train_loader = torch.utils.data.DataLoader(sub_dset_train, batch_size=20, shuffle=True, num_workers=4)

        sub_dset_test = DatasetProcessing(sub_X_test, sub_Y_test)
        sub_test_loader = torch.utils.data.DataLoader(sub_dset_test, batch_size=20, shuffle=False, num_workers=0)

        train_loader_list.append(sub_train_loader)
        test_loader_list.append(sub_test_loader)
        dset_train_list.append(sub_dset_train)

    return train_loader_list, test_loader_list, dset_train_list


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
        "- areas: {}".format(args.areas) + "\n",
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
        







