import numpy as np
import pandas as pd
import math
import torch
import torchvision
import matplotlib.pyplot as plt
import os

from datetime import datetime
from PIL import Image
from keras.datasets import cifar10
from keras import backend as K  
K.set_image_data_format('channels_last') 



def create_loader_list(args):
    
    if args.source == 'torchvision':
    
        transform_cifar10_train = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(0.5),
                                                              torchvision.transforms.RandomChoice([torchvision.transforms.RandomRotation(10), torchvision.transforms.RandomCrop(size=[32,32], padding=4, padding_mode='edge')]),
                                                              #torchvision.transforms.RandomCrop(size=[32,32], padding=4, padding_mode='edge'),
                                                              torchvision.transforms.ToTensor(), 
                                                              torchvision.transforms.Normalize(mean=(0.0,), std=(1.0,))]) 

        transform_cifar10_test = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                                             torchvision.transforms.Normalize(mean=(0.0,), std=(1.0,))]) 


        cifar10_train_dset = torchvision.datasets.CIFAR10('./cifar10_pytorch', train=True, transform=transform_cifar10_train, download=True)
        dataset_length = len(cifar10_train_dset)
    
        cifar10_test_dset = torchvision.datasets.CIFAR10('./cifar10_pytorch', train=False, transform=transform_cifar10_test, download=True)
        cifar10_test_loader = torch.utils.data.DataLoader(cifar10_test_dset, batch_size=256, shuffle=False, num_workers=1)

        print('train set :', cifar10_train_dset)
        print('train transform :', cifar10_train_dset.transform)
        print('test set :', cifar10_test_dset)
        print('test transform :', cifar10_test_dset.transform)

        train_loader_list = []
        test_loader_list = [cifar10_test_loader]

        for i in range(args.nb_subset):
    
            train_loader_list.append(torch.utils.data.DataLoader(cifar10_train_dset, batch_size=args.mbs, 
                                                   sampler = torch.utils.data.SubsetRandomSampler(range(int(i*(dataset_length/args.nb_subset)),int((i+1)*(dataset_length/args.nb_subset)))), 
                                                   shuffle=False, num_workers=1))
        
        full_train_loader = torch.utils.data.DataLoader(cifar10_train_dset, batch_size=args.mbs, shuffle=True, num_workers=1)

    elif args.source == 'keras':
        
        dataset_length = 50000
        permut = np.random.permutation(dataset_length)

        transform_cifar10_train = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(), torchvision.transforms.RandomHorizontalFlip(0.5),
                                                          torchvision.transforms.RandomChoice([torchvision.transforms.RandomRotation(10), torchvision.transforms.RandomCrop(size=[32,32], padding=4, padding_mode='edge')]),
                                                          #torchvision.transforms.RandomCrop(size=[32,32], padding=4, padding_mode='edge'),
                                                          torchvision.transforms.ToTensor(), 
                                                          torchvision.transforms.Normalize(mean=(0.0,), std=(1.0,))]) 

        transform_cifar10_test = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(), torchvision.transforms.ToTensor(), 
                                                         torchvision.transforms.Normalize(mean=(0.0,), std=(1.0,))]) 


        (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

        dset_train = DatasetProcessing(X_train[permut,:,:,:], Y_train[permut], transform_cifar10_train)
        full_train_loader = torch.utils.data.DataLoader(dset_train, batch_size=args.mbs, shuffle=True, num_workers=1)

        dset_test = DatasetProcessing(X_test, Y_test, transform_cifar10_test)
        test_loader = torch.utils.data.DataLoader(dset_test, batch_size=args.mbs, shuffle=True, num_workers=1)

        train_loader_list = []
        test_loader_list = [test_loader]

        for i in range(args.nb_subset):
            train_loader_list.append(torch.utils.data.DataLoader(dset_train, batch_size=args.mbs, 
                                                   sampler = torch.utils.data.SubsetRandomSampler(range(int(i*(dataset_length/args.nb_subset)),int((i+1)*(dataset_length/args.nb_subset)))), 
                                                   shuffle=False, num_workers=1))

    return train_loader_list, test_loader_list, full_train_loader


class DatasetProcessing(torch.utils.data.Dataset): #Data must be wrapped on a Dataset parent class where the methods __getitem__ and __len__ must be overrided. Note that,the data is not loaded on memory by now.
    
    def __init__(self, data, target, transform=None): #used to initialise the class variables - transform, data, target
        
        self.transform = transform
        self.data = data.astype(np.uint8)
        self.target = torch.from_numpy(np.squeeze(target)).long() # needs to be in torch.LongTensor dtype
   
    def __getitem__(self, index): #used to retrieve the X and y index value and return it
        
        if self.transform is not None:
            return self.transform(self.data[index]), self.target[index]
        else:
            return self.data[index], self.target[index]
    
    def __len__(self): #returns the length of the data
        return len(list(self.data))



def plot_acc(path, data):
    df = pd.read_csv(path+'/'+data)
    x = df['epoch']
    y = df['acc_test']
    plt.figure(figsize=(14,12))

    plt.plot(x,y)
    plt.ylim((40,100))
    plt.yticks([40,60,80,90,100])
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (\%)')
    plt.grid()
    plt.savefig(path + '/test_acc.png', format = 'png')
    plt.close()

def createHyperparametersFile(path, args):

    hyperparameters = open(path + r"/hyperparameters.txt","w+")
    L = ["- source: {}".format(args.source) + "\n",
        "- lr: {}".format(args.lr) + "\n",
        "- MiniBatchSize: {}".format(args.mbs) + "\n",
        "- Meta: {}".format(args.meta) + "\n",
        "- Weight decay: {}".format(args.weight_decay) + "\n",
        "- epochs per task: {}".format(args.epochs_per_task) + "\n",
        "- init: {}".format(args.init) + "\n",
        "- init width: {}".format(args.init_width) + "\n",
        "- nb_subset: {}".format(args.nb_subset) + "\n"]
   
    hyperparameters.writelines(L)
    hyperparameters.close()

