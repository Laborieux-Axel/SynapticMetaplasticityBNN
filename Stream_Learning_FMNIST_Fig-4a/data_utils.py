import numpy as np
import math
import pandas as pd
import torch
import torchvision
import matplotlib.pyplot as plt
import os
import argparse

from collections import OrderedDict
from datetime import datetime


class DatasetProcessing(torch.utils.data.Dataset): #Data must be wrapped on a Dataset parent class where the methods __getitem__ and __len__ must be overrided. Note that,the data is not loaded on memory by now.
    
    def __init__(self, data, target, transform=None): #used to initialise the class variables - transform, data, target
        self.transform = transform
        
        if data.shape[1] == 28:   # in this case data is a mnist like dataset 
            self.data = data.astype(np.uint8)[:,:,:,None]  # .reshape((-1,28,28))
        else:  # in this case data is other embedded tasks (cifar, kmnist...)
            self.data = data.astype(np.float32)[:,:,None] 
        print(target.shape)
        self.target = torch.from_numpy(target).long() # needs to be in torch.LongTensor dtype
   
    def __getitem__(self, index): #used to retrieve the X and y index value and return it
        
        if self.transform is not None:
            return self.transform(self.data[index]), self.target[index]
        else:
            return self.data[index], self.target[index]
    
    def __len__(self): #returns the length of the data
        return len(list(self.data))


def createHyperparametersFile(path, args):

    hyperparameters = open(path + r"/hyperparameters.txt","w+")
    L = ["- task: {}".format(args.task) + "\n",
        "- lr: {}".format(args.lr) + "\n",
        "- MiniBatchSize: {}".format(args.mbs) + "\n",
        "- Meta: {}".format(args.meta) + "\n",
        "- Weight decay: {}".format(args.decay) + "\n",
        "- beaker: {}".format(args.beaker) + "\n",
        "- number of beakers: {}".format(args.n_bk) + "\n",
        "- ratios: {}".format(args.ratios) + "\n",
        "- feedback: {}".format(args.fb) + "\n",
        "- epochs per task: {}".format(args.epochs_per_task) + "\n",
        "- init: {}".format(args.init) + "\n",
        "- init width: {}".format(args.init_width) + "\n",
        "- nb_subset: {}".format(args.nb_subset) + "\n"]
   
    hyperparameters.writelines(L)
    hyperparameters.close()


