import numpy as np
import math
import pandas as pd
import torch
import torchvision
import matplotlib
import matplotlib.pyplot as plt
import os
import argparse

assert(torch.__version__ <= '1.1.0')

matplotlib.use('Agg')

os.environ['KMP_WARNINGS'] = 'off'

from collections import OrderedDict
from datetime import datetime
from model_utils import *
from data_utils import *
from keras.datasets import fashion_mnist, mnist


parser = argparse.ArgumentParser(description='BNN learning sequentially fraction of the same dataset , metaplasticity is controlled by the argument meta.')
parser.add_argument('--hidden-layers', nargs = '+',type = int,default = [1024,1024], metavar = 'HL',help='size of the hidden layers')
parser.add_argument('--lr',type = float,default = 0.005, metavar = 'LR',help='Learning rate')
parser.add_argument('--epochs-per-task',type = int,default = 5, metavar = 'EPT',help='Number of epochs per tasks')
parser.add_argument('--mbs',type = int,default = 100, metavar = 'MBS',help='Mini batch size')
parser.add_argument('--nb-subset',type = int,default = 1, metavar = 'SS',help='Number of partitions')
parser.add_argument('--meta',type = float,default = 0.0, metavar = 'M',help='Metaplasticity coefficient')
parser.add_argument('--decay',type = float, default = 0.0, metavar = 'dc',help='Weight decay')
parser.add_argument('--init',type = str,default = 'uniform', metavar = 'INIT',help='Weight initialisation')
parser.add_argument('--init-width',type = float,default = 0.1, metavar = 'W',help='Weight initialisation width')
parser.add_argument('--task',type = str,default = 'FMNIST', metavar = 'T',help='Task to choose between FMNIST and MNIST')
parser.add_argument('--save',type = bool,default = True, metavar = 'S',help='Saving the results')
parser.add_argument('--device',type = int,default = 0, metavar = 'Dev',help='Device on which run the simulation')
parser.add_argument('--beaker', default = False, action = 'store_true', help='use beaker')
parser.add_argument('--fb', type = float, default = 5e-3, metavar = 'fb', help='feeback coeff from last beaker to the first')
parser.add_argument('--n-bk', type = int, default = 4, metavar = 'bk', help='number of beakers')
parser.add_argument('--ratios', nargs = '+', type = float, default = [1e-2,1e-3,1e-4,1e-5], metavar = 'Ra', help='pipes specs between beakers')



args = parser.parse_args()

device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")

date = datetime.now().strftime('%Y-%m-%d')
time = datetime.now().strftime('%H-%M-%S')
path = 'results/'+date+'/'+time+'_gpu'+str(args.device)
if not(os.path.exists(path)):
    os.makedirs(path)

createHyperparametersFile(path, args)

# Hyperparameters
lr = args.lr
epochs = args.epochs_per_task
save_result = args.save
meta = args.meta
archi = [784]+args.hidden_layers+[10]
nb_subset = args.nb_subset
task_names = ['sub_'+args.task+'_']
mbs = args.mbs
dataset_length = 60000
permut = np.random.permutation(dataset_length)

# Data preparation
transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(), 
                                            torchvision.transforms.ToTensor(), 
                                            torchvision.transforms.Normalize(mean=(0.0,), std=(1.0,))])

if args.task == 'FMNIST':
    (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
elif args.task == 'MNIST':
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

dset_train = DatasetProcessing(X_train[permut,:,:], Y_train[permut], transform)
train_loader = torch.utils.data.DataLoader(dset_train, batch_size=mbs, shuffle=True, num_workers=1)

dset_test = DatasetProcessing(X_test, Y_test, transform)
test_loader = torch.utils.data.DataLoader(dset_test, batch_size=mbs, shuffle=True, num_workers=1)

train_loader_list = []

for i in range(nb_subset):
    train_loader_list.append(torch.utils.data.DataLoader(dset_train, batch_size=mbs, 
                                                   sampler = torch.utils.data.SubsetRandomSampler(range(int(i*(dataset_length/nb_subset)),int((i+1)*(dataset_length/nb_subset)))), 
                                                   shuffle=False, num_workers=1))


model = BNN(archi, init = args.init, width = args.init_width).to(device)

# plot_parameters(model, path, save=save_result)

# Results collect initialisation
data = {}
data['net'] = 'bnn'
arch = ''
for i in range(model.hidden_layers):
    arch = arch + '-' + str(model.layers_dims[i+1])

data['arch'] = arch[1:]
data['lr'], data['meta'], data['init'], data['task_order'], data['tsk'], data['epoch'], data['acc_tr_sub'], data['acc_tr'] = [], [], [], [], [], [], [], []
data['acc_test'], data['acc_1st_sub'] = [], []

name = '_'+data['net']+'_'+data['arch']+'_'+args.task

if not(args.beaker):
    optimizer = Adam_meta(model.parameters(), lr = lr, meta = meta, weight_decay = args.decay)
else:
    optimizer = Adam_bk(model.parameters(), lr = lr, n_bk=args.n_bk, ratios=args.ratios, feedback=args.fb, meta=meta, weight_decay=args.decay, path=path)

for task_idx, task in enumerate(train_loader_list):

    for epoch in range(1, epochs+1):

        train(model, task, task_idx, optimizer, device) 
         
        data['init'].append(args.init_width)
        data['meta'].append(meta)
        data['task_order'].append(task_idx+1)
        data['tsk'].append(task_names[0]+str(task_idx+1)+'/'+str(nb_subset))
        data['epoch'].append(epoch)
        data['lr'].append(lr)
        data['acc_tr_sub'].append(test(model, task, device, frac = nb_subset))
        
        data['acc_1st_sub'].append(test(model, train_loader_list[0], device, frac = nb_subset))

        train_accuracy = test(model, train_loader, device)
        data['acc_tr'].append(train_accuracy)

        test_accuracy = test(model, test_loader, device)
        data['acc_test'].append(test_accuracy)
        
    #plot_parameters(model, path, save=save_result)
    df_data = pd.DataFrame(data)
    if save_result:
        time = datetime.now().strftime('%H-%M-%S') 
        df_data.to_csv(path +'/'+time+name+str(task_idx+1)+'-'+str(nb_subset)+'.csv', index = False)


